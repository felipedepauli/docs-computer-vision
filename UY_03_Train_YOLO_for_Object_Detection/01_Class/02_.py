# import cv2

# # Instead of using a trackbar, we will use a static value
# # to control the lower and upper boundaries of the HSV pixel
# min_blue, min_green, min_red = 21, 222, 70
# max_blue, max_green, max_red = 176, 255, 255

# # Defining object for reading video from camera
# camera = cv2.VideoCapture(0)

# # Check if the camera is opened successfully
# if not camera.isOpened():
#     print("Error: Unable to access the camera.")
#     exit()

# # Defining loop for catching frames
# while True:
#     # Capture frame-by-frame from camera
#     ret, frame_BGR = camera.read()

#     if not ret:
#         print("Error: Unable to read the frame.")
#         break

#     # Converting current frame to HSV
#     frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

#     # Implementing Mask with founded colors from Track Bars to HSV Image
#     mask = cv2.inRange(frame_HSV, (min_blue, min_green, min_red), (max_blue, max_green, max_red))

#     # Show the frames
#     cv2.imshow('BGR Frame', frame_BGR)
#     cv2.imshow('HSV Frame', frame_HSV)
#     cv2.imshow('Masked Frame', mask)

#     # Break the loop if the user presses 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the windows
# camera.release()
# cv2.destroyAllWindows()

# print("wow")



# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()
import cv2 as cv

print(cv.__version__)

import numpy as np
cap = cv.VideoCapture(0, cv.CAP_V4L2)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()