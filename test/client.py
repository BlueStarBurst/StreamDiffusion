# import libraries
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
import cv2

#define netgear client with `receive_mode = True` and default settings
client = NetGear(port=3000,receive_mode = True)
server = NetGear(port=3001) #Define netgear server with default settings
stream = VideoGear(resolution=(1920,1080)).start() #Open any video stream

cv2.namedWindow("Output Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output Frame", 1920, 1080)

# infinite loop
while True:
    try: 
        frame = stream.read()
        # read frames

        # check if frame is None
        if frame is None:
            #if True break the infinite loop
            break

        # do something with frame here

        # send frame to server
        server.send(frame)
    
    except KeyboardInterrupt:
        #break the infinite loop
        break
    
    # receive frames from network
    frame = client.recv()

    # check if frame is None
    if frame is None:
        #if True break the infinite loop
        break

    # do something with frame here

    # Show output window
    frame = cv2.resize(frame, (1920, 1080))
    cv2.imshow("Output Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

# close output window
cv2.destroyAllWindows()
# safely close client
client.close()