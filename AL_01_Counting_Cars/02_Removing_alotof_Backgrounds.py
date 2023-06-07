import numpy as np
import cv2

video0 = "..\common\media\\videos\Arco.mp4"
video1 = "..\common\media\\videos\Estradas.mp4"
video2 = "..\common\media\\videos\Peixes.mp4"
video3 = "..\common\media\\videos\Ponte.mp4"
video4 = "..\common\media\\videos\Rua.mp4"

cap0 = cv2.VideoCapture(video0)
cap1 = cv2.VideoCapture(video1)
cap2 = cv2.VideoCapture(video2)
cap3 = cv2.VideoCapture(video3)
cap4 = cv2.VideoCapture(video4)

hasFrame0, frame0 = cap0.read()
hasFrame1, frame1 = cap1.read()
hasFrame2, frame2 = cap2.read()
hasFrame3, frame3 = cap3.read()
hasFrame4, frame4 = cap4.read()

framesIds0 = cap0.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)
framesIds1 = cap1.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)
framesIds2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)
framesIds3 = cap3.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)
framesIds4 = cap4.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=75)

frames0 = []
frames1 = []
frames2 = []
frames3 = []
frames4 = []

print("Getting the random frames 0...")
for fid in framesIds0:
    cap0.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame0, frame0 = cap0.read()
    frames0.append(frame0)
    
print("Getting the random frames 1...")
for fid in framesIds1:
    cap1.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame1, frame1 = cap1.read()
    frames1.append(frame1)
    
print("Getting the random frames 2...")
for fid in framesIds2:
    cap2.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame2, frame2 = cap2.read()
    frames2.append(frame2)
    
print("Getting the random frames 3...")
for fid in framesIds3:
    cap3.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame3, frame3 = cap3.read()
    frames3.append(frame3)
    
print("Getting the random frames 4...")
for fid in framesIds4:
    cap4.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame4, frame4 = cap4.read()
    frames4.append(frame4)
    
print("Calculating the median frame on frame 0...")
medianFrame0 = np.median(frames0, axis=0).astype(dtype=np.uint8)
print("Calculating the median frame on frame 1...")
medianFrame1 = np.median(frames1, axis=0).astype(dtype=np.uint8)
print("Calculating the median frame on frame 2...")
medianFrame2 = np.median(frames2, axis=0).astype(dtype=np.uint8)
print("Calculating the median frame on frame 3...")
medianFrame3 = np.median(frames3, axis=0).astype(dtype=np.uint8)
print("Calculating the median frame on frame 4...")
medianFrame4 = np.median(frames4, axis=0).astype(dtype=np.uint8)
print("Done!")

cv2.imshow('Median Frame 0', medianFrame0)
cv2.imshow('Median Frame 1', medianFrame1)
cv2.imshow('Median Frame 2', medianFrame2)
cv2.imshow('Median Frame 3', medianFrame3)
cv2.imshow('Median Frame 4', medianFrame4)

cv2.waitKey(0)