import cv2
import numpy as np
import sys

video = "../common/media/videos/Ponte.mp4"

cap = cv2.VideoCapture(video)
    
algorithm_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']
algorithm_type = algorithm_types[2]

# KNN  = 9.8644371   9.6577013   9.8870235  
# GMG  = 20.6283763 20.7968725  20.9903735
# CNT  = 5.1683298   4.8186288   4.8525018  Faster!
# MOG  = 14.579084   15.442477  15.5900635
# MOG2 = 9.6979276   9.0108875   8.7852331


# First, we define the Kernels
def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilatation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3),np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3,3),np.uint8)
    if KERNEL_TYPE == 'erosion':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    return kernel

# Then, we apply them to the image
def Filter(image, filter):
    if filter == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN,  Kernel('opening'), iterations=2)
    if filter == 'dilatation':
        return cv2.dilate(image, Kernel('dilatation'), iterations=2)
    # This combines all the three filters. There is a big difference in the computation cost
    if filter == 'combine':
        closing  = cv2.morphologyEx(image, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening  = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilatation'), iterations=2)
        return dilation

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
background_subtractor = Subtractor('MOG2')

if __name__ == "__main__":
    
    while cap.isOpened:
        
        ok, frame = cap.read()
        
        if not ok:
            print('Frames ended')
            break
        
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        
        mask = background_subtractor.apply(frame)
        mask_filtered = Filter(mask, 'combine')
        
        cars_after_mask = cv2.bitwise_and(frame, frame, mask=mask_filtered)
        
        
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Mask Filtered', mask_filtered)
        cv2.imshow('Cars after Mask', cars_after_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened:
        ok, frame = cap.read()
        
        if not ok:
            print('Frames ended')
            break
        
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        mask = background_subtractor.apply(frame)

        
        mask_Filter_closing  = Filter(mask, 'closing')
        mask_Filter_opening  = Filter(mask, 'opening')
        mask_Filter_combine  = Filter(mask, 'combine')
        

        cv2.imshow('Mask Filter Closing',  mask_Filter_closing)
        cv2.imshow('Mask Filter Opening',  mask_Filter_opening)
        cv2.imshow('Mask Filter Combine',  mask_Filter_combine)
        
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.closeAllWindows()
cap.release()