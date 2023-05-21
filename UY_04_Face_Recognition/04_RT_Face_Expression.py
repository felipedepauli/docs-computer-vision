import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

# Face expression model initialization
face_exp_model = model_from_json(open("models/facial_expression_model_structure.json", "r").read())


# Load weights into the model


# List of emotion labels
