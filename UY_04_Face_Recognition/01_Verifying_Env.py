import dlib
import cv2
import face_recognition

print("GPU enabled    =", dlib.DLIB_USE_CUDA)
print("Dlib version   =", dlib.__version__)
print("OpenCV version =", cv2.__version__)
print("Face_recognition version =", face_recognition.__version__)