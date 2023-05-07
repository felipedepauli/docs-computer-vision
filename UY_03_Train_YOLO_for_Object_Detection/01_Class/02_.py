"""
Course: Training YOLO v3 for Objects Detection with Custom Data

Section-1
Quick Win - Step 2: Simple Object Detection by thresholding with mask
File: detecting-object.py
"""


# Who's ready for some object detection magic? Let's get started!
# 
# Algorithm:
# Reading RGB image --> Converting to HSV --> Implementing Mask -->
# --> Finding Contour Points --> Extracting Rectangle Coordinates -->
# --> Drawing Bounding Box --> Putting Label
#
# Result:
# Window with Detected Object, Bounding Box and Label in Real Time


# Importing the magical library that makes all of this possible: OpenCV
import cv2
import numpy as np

# Defining lower bounds and upper bounds of the mask for our mysterious object
# Defining the color bounds
color_bounds = {
    "blue": {
        "lower": np.array([100, 150, 0]),
        "upper": np.array([140, 255, 255])
    },
    "red": {
        "lower": np.array([0, 150, 50]),
        "upper": np.array([10, 255, 255]),
        "lower2": np.array([170, 150, 50]),
        "upper2": np.array([180, 255, 255])
    },
    "green": {
        "lower": np.array([40, 50, 50]),
        "upper": np.array([90, 255, 255])
    },
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255])
    },
    "pink": {
        "lower": np.array([140, 75, 75]),
        "upper": np.array([170, 255, 255])
    }
}

# Choose the color you want to detect by setting the color_name variable
color_name = "red"

# Time to find out which version of OpenCV we're using!
# Split the version string by dots and grab the first part
v = cv2.__version__.split('.')[0]

# Let's create an object for reading video from the camera
camera = cv2.VideoCapture(0)
# Ps. you cannot be in wsl to use the camera

# Loop time! Catching frames and making magic happen
while True:
    # Capture frame-by-frame from the camera
    a, frame_BGR = camera.read()
    print(a)

    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # Create the mask for the chosen color
    if color_name == "red":
        mask1 = cv2.inRange(frame_HSV, color_bounds[color_name]["lower"], color_bounds[color_name]["upper"])
        mask2 = cv2.inRange(frame_HSV, color_bounds[color_name]["lower2"], color_bounds[color_name]["upper2"])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(frame_HSV, color_bounds[color_name]["lower"], color_bounds[color_name]["upper"])

    # # Convert the current frame to HSV color space
    # frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # # Let's make a mask for our object using its color bounds in the HSV space
    # mask = cv2.inRange(frame_HSV,
    #                    (min_blue, min_green, min_red),
    #                    (max_blue, max_green, max_red))

    # Show the frame with the mask applied
    cv2.namedWindow('Binary frame with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary frame with Mask', mask)

    # Finding contours - the outline of our object
    # Different OpenCV versions return different numbers of parameters
    # when using cv2.findContours()

    # If we're using OpenCV version 3
    if v == '3':
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # If we're using OpenCV version 4
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Let's find the biggest contour (our object) by sorting them by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # If we found any contours, let's extract their coordinates
    if contours:
        # Get the bounding rectangle around the largest contour
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Draw the bounding box on the current BGR frame
        cv2.rectangle(frame_BGR, (x_min - 15, y_min - 15),
                      (x_min + box_width + 15, y_min + box_height + 15),
                      (0, 255, 0), 3)

        # Prepare the label text
        label = 'Detected Object'

        # Put the label on the current BGR frame
        cv2.putText(frame_BGR, label, (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show the current BGR frame with the detected object, bounding box, and label
    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', frame_BGR)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all the opened windows to clean up our workspace
cv2.destroyAllWindows()


"""
Some comments

With OpenCV function cv2.findContours() we find 
contours of white object from black background.

There are three arguments in cv.findContours() function,
first one is source image, second is contour retrieval mode,
third is contour approximation method.

In OpenCV version 3, the cv2.findContours() function returns three parameters:
modified image, the contours, and hierarchy.
Further reading about Contours in OpenCV v3:
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html

In OpenCV version 4, the cv2.findContours() function returns two parameters:
the contours and hierarchy.
Further reading about Contours in OpenCV v4:
https://docs.opencv.org/4.0.0/d4/d73/tutorial_py_contours_begin.html

Contours is a Python list of all the contours in the image.
Each individual contour is a Numpy array of (x, y) coordinates 
of boundary points of the object.

Contours can be explained simply as a curve joining all the 
continuous points (along the boundary), having the same color or intensity.
"""

