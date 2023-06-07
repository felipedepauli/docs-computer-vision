import cv2
import numpy as np
import sys
import csv

video = "../common/media/videos/Ponte.mp4"

cap = cv2.VideoCapture(video)

fp = open('Results.csv', mode = 'w')
writer = csv.DictWriter(fp, fieldnames = ['Frame', 'Pixel Count'])
writer.writeheader()

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
background_subtractor = []

for i, algorithm_type in enumerate(algorithm_types):
    print(i, algorithm_type)
    background_subtractor.append(Subtractor(algorithm_type))

if __name__ == "__main__":
    
    while cap.isOpened:
        
        frame = cap.read()[1]
        
        frame = cv2.resize(frame, (320, 240))
        
        knn  = background_subtractor[0].apply(frame)
        gmg  = background_subtractor[1].apply(frame)
        cnt  = background_subtractor[2].apply(frame)
        mog  = background_subtractor[3].apply(frame)
        mog2 = background_subtractor[4].apply(frame)
        
        knnCount  = np.count_nonzero(knn)
        gmgCount  = np.count_nonzero(gmg)
        cntCount  = np.count_nonzero(cnt)
        mogCount  = np.count_nonzero(mog)
        mog2Count = np.count_nonzero(mog2)
        
        writer.writerow({'Frame': 'KNN',  'Pixel Count': knnCount})
        writer.writerow({'Frame': 'GMG',  'Pixel Count': gmgCount})
        writer.writerow({'Frame': 'CNT',  'Pixel Count': cntCount})
        writer.writerow({'Frame': 'MOG',  'Pixel Count': mogCount})
        writer.writerow({'Frame': 'MOG2', 'Pixel Count': mog2Count})
        
        cv2.imshow('Frame', frame)
        cv2.imshow('KNN', knn)
        cv2.imshow('GMG', gmg)
        cv2.imshow('CNT', cnt)
        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break