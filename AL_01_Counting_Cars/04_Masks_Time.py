import cv2
import numpy as np
import sys

video = "../common/media/videos/Ponte.mp4"

cap = cv2.VideoCapture(video)

# delay = int((1/30*1000))

# while True:
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         print("No frame")
#         break
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(delay) & 0xFF == ord('q'):
#         break
    
    
algorithm_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']
algorithm_type = algorithm_types[2]

# KNN  = 9.8644371   9.6577013   9.8870235  
# GMG  = 20.6283763 20.7968725  20.9903735
# CNT  = 5.1683298   4.8186288   4.8525018  Faster!
# MOG  = 14.579084   15.442477  15.5900635
# MOG2 = 9.6979276   9.0108875   8.7852331




def Subtractor(algorithm_type):
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    print('Erro - Insira uma nova informação')
    sys.exit(1)
    
cap = cv2.VideoCapture(video)
background_subtractor = Subtractor(algorithm_type)


if __name__ == "__main__":
    e1 = cv2.getTickCount()
    frame_number = -1
    while cap.isOpened:
        ok, frame = cap.read()
        
        frame_number += 1
        
        if not ok:
            print('Frames over')
            break
        
        mask = background_subtractor.apply(frame)
        # cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or frame_number > 300:
            break
        
    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    print(t)
         