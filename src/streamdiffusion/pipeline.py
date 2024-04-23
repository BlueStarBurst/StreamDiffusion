import time
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers import AsymmetricAutoencoderKL, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

# from ...image_processor import PipelineImageInput, VaeImageProcessor


from streamdiffusion.image_filter import SimilarImageFilter

from PIL import Image


class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionInpaintPipeline,
        # vae: Union[AutoencoderKL, AsymmetricAutoencoderKL],
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height = int(height // pipe.vae_scale_factor)
        self.latent_width = int(width // pipe.vae_scale_factor)
        # self.vae_scale_factor = pipe.vae_scale_factor

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        self.similar_image_filter = False
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None

        self.pipe = pipe

        self.vae_scale_factor = 2 ** (
            len(self.pipe.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=pipe.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        self.inference_time_ema = 0

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        # initialize x_t_latent (it can be any random tensor)
        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        if self.use_denoising_batch and self.cfg_type == "full":
            uncond_prompt_embeds = encoder_output[1].repeat(
                self.batch_size, 1, 1)
        elif self.cfg_type == "initialize":
            uncond_prompt_embeds = encoder_output[1].repeat(
                self.frame_bff_size, 1, 1)

        if self.guidance_scale > 1.0 and (
            self.cfg_type == "initialize" or self.cfg_type == "full"
        ):
            self.prompt_embeds = torch.cat(
                [uncond_prompt_embeds, self.prompt_embeds], dim=0
            )

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # make sub timesteps list based on the indices in the t_list list and the values in the timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self.stock_noise = torch.zeros_like(self.init_noise)

        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in self.sub_timesteps:
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (
                1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor,
                        width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        # If the masked_image is already in latents space, we don't need to encode it again
        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            # encode the masked image to latents
            masked_image_latents = self._encode_vae_image(
                masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # Do classifier free guidance if needed
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] *
                      2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(
            device=device, dtype=dtype)
        return mask, masked_image_latents

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch -
                self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        mask_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat(
                [x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        model_pred = None
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        latent_model_input = x_t_latent_plus_uc

        if mask is not None:
            print(num_channels_latents)
            print(num_channels_unet)
            print(mask.shape)
            print(mask_latent.shape)
            # 8. Check that sizes of mask, masked image and latents match
            if num_channels_unet == 9:
                # default case for runwayml/stable-diffusion-inpainting
                num_channels_mask = mask.shape[1]
                num_channels_masked_image = mask_latent.shape[1]
                if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                    raise ValueError(
                        f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                        f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input."
                    )
            elif num_channels_unet != 4:
                raise ValueError(
                    f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
                )
            
            
            # latent_model_input = torch.cat([x_t_latent_plus_uc] * 2) if self.do_classifier_free_guidance else x_t_latent_plus_uc
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_list)
            print(latent_model_input.size(), mask.size(), mask_latent.size())
                
            if num_channels_unet == 9:
                latent_model_input = torch.cat(
                    [latent_model_input, mask, mask_latent], dim=1)
            
            # print("Look Here:")
            # print(latent_model_input.size(), mask.size(), mask_latent.size())
            
            # latent_model_input = torch.cat(
            #     [latent_model_input, mask, mask_latent], dim=1)

            model_pred = self.unet(
                latent_model_input,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0]
        else:
            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                return_dict=False,
            )[0]

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(
                model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(
                    model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(
                model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(
            self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent

    def predict_x0_batch(
        self,
        x_t_latent: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # copy the x_t_latent to the buffer if the buffer is None
        # self.x_t_latent_buffer = x_t_latent.clone()
        
        if self.x_t_latent_buffer is None:
            self.x_t_latent_buffer = x_t_latent
        
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
            x_0_pred_batch, model_pred = self.unet_step(
                x_t_latent, t_list, mask=mask, mask_latent=mask_latent)

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
                # latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                
                # duplicate a layer of the mask to be 3 dimensional (currently [2, 1, 64, 64]) to match the size of the latents
                mask = mask[0].repeat(3, 4, 1, 1)
                print(mask[0][0][0][0])
                print(mask[1][0][0][0])
                print(prev_latent_batch[0][0][0][0])
                print(prev_latent_batch[1][0][0][0])
                
                # expand mask dim 0 to 3 to match the size of the latents
                
                
                print(mask.size(), prev_latent_batch.size())
                self.x_t_latent_buffer = (mask) * self.x_t_latent_buffer + (1 - mask) * prev_latent_batch
            else:
                x_0_pred_out = x_0_pred_batch
                
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            self.x_t_latent_buffer = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(x_t_latent, t, idx, mask, mask_latent)
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
                        
                # latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                x_t_latent = (1 - mask) * x_t_latent + mask * prev_latent_batch
                self.x_t_latent_buffer = x_t_latent
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert(
                "RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(
                    image[i: i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @torch.no_grad()
    def __call__(
        self,
        x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None,
        # image: PipelineImageInput = None,
        prompt: Union[str, List[str]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 7.5,
        clip_skip: Optional[int] = None,
    ) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # set default values to the input parameters
        batch_size = 1
        crops_coords = None
        # crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
        resize_mode = "default"
        device = self.pipe._execution_device
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip

        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        # latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # Set image and init_image
        # original_image = image
        # init_image = self.image_processor.preprocess(
        #     image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        # )
        # init_image = init_image.to(dtype=torch.float32)

        # encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get(
                "scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=self.prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        # Set regular latents
        # latents_outputs = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        #     image=init_image,
        #     timestep=latent_timestep,
        #     is_strength_max=is_strength_max,
        #     return_noise=True,
        #     return_image_latents=return_image_latents,
        # )

        mask_condition = self.pipe.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        mask_condition = mask_condition.to(dtype=torch.float32)

        # create mask_latent if mask is not None
        mask_latent = None
        if mask is not None:
            mask_latent = retrieve_latents(
                self.vae.encode(mask), self.generator)
            mask_latent = mask_latent * self.vae.config.scaling_factor

        init_image = self.image_processor.preprocess(x, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32, device="cpu")

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents
            
        if height is None:
            height = self.height
        if width is None:
            width = self.width

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            self.height,
            self.width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        if x is not None:
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
            x_t_latent = self.encode_image(x)
        else:
            # TODO: check the dimension of x_t_latent

            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )

        if mask is not None:
            x_0_pred_out = self.predict_x0_batch(
                x_t_latent, mask=mask, mask_latent=masked_image_latents)
        else:
            x_0_pred_out = self.predict_x0_batch(x_t_latent)
        x_output = self.decode_image(x_0_pred_out).detach().clone()

        self.prev_image_result = x_output
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end) / 1000
        self.inference_time_ema = 0.9 * self.inference_time_ema + 0.1 * inference_time
        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        )
        x_output = self.decode_image(x_0_pred_out).detach().clone()
        return x_output

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)
