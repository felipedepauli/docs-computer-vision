import numpy as np
import cv2

streetVideo = "..\common\media\\videos\Rua.mp4"

# First, we have to create the Capture object
cap = cv2.VideoCapture(streetVideo)
hasFrame, frame = cap.read()

# Then, we are going to remove the background
# We need to get 75 frames from the video randomly
# (we could get all the frames, but it would take a lot of time)
# and then calculate the median frame
framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)

frames = []

for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame = cap.read()
    frames.append(frame)
    
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Now we have the median frame, we can remove the background
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

cv2.imwrite('medianFrame.png', medianFrame)

# We are going to convert the median frame to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

cv2.imwrite('medianFrameGray.png', grayMedianFrame)


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while (True):
    hasFrame, frame = cap.read()
    
    # When the camera is off, the hasFrame is False (or the video is over)
    if (not hasFrame):
        break
    
    # We convert it to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # We calculate the difference between the median frame and the current frame
    dFrame = cv2.absdiff(grayMedianFrame, frameGray)
    
    # We apply a threshold to the difference
    # If the difference is greater than 30, we set it to 255
    # If the difference is less than 30, we set it to 0
    # This way, we can remove the background
    # th, dFrame = cv2.threshold(dFrame, 30, 255, cv2.THRESH_BINARY)
    th, dFrame = cv2.threshold(dFrame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    
    cv2.imshow('frameGray', dFrame)
    if (cv2.waitKey(100) & 0xFF == ord('q')):
        break
    
cap.release()
    
