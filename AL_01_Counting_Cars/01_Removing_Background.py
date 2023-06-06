import numpy as np
import cv2

# Path of the video we want to remove the background
video = "..\common\media\demo.mp4"

# Read the video and get the first frame
cap = cv2.VideoCapture(video)
hasFrame, frame = cap.read()

# We can show it using imshow
cv2.imshow('frame', frame)
cv2.waitKey(0)

# We can show 10 frames using a for loop
for i in range(10):
    hasFrame, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    
print( "Initializing the background remove...")
    
# Now we get 75 frames from the total frames
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
framesIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)

# With the Ids, we can get the frames
frames = []
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid) # Set the next frame id we're goingo to get (CAP_PROP_POS_FRAMES is the property)
    hasFrame, frame = cap.read()          # Read the frame
    frames.append(frame)                  # Append the frame to the list

# Now we can calculate the median frame
# The median is the middle value of a sorted list of numbers
# We can use the median to remove the background
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
print(medianFrame)

# We can show the median frame
cv2.imshow('Median Frame', medianFrame)
cv2.waitKey(0)