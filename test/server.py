# import libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import torch

from img2img import Pipeline

from config import config, Args

import sys
sys.path.append('../')

from utils.wrapper import StreamDiffusionWrapper

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

args = config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16
pipeline = Pipeline(config, device, torch_dtype)


server = NetGear(port=3000) #Define netgear server with default settings
client = NetGear(port=3001, receive_mode = True)

# infinite loop until [Ctrl+C] is pressed
while True:
    
    # receive frames from network
    frame = client.recv()

    # check if frame is None
    if frame is None:
        #if True break the infinite loop
        break
    
    try: 
        params = pipeline.InputParams(image=frame, prompt="an anime boy", width=512, height=512)
        image = pipeline.predict(params)
        server.send(image)
    
    except KeyboardInterrupt:
        #break the infinite loop
        break

# safely close server
server.close()