'''
---------------------------------
    @  COURSE NAME
        Course:  Training YOLO v3 for Objects Detection with Custom Data

    @  SECTION
        Quick Win - Step 1: Simple Object Detection by thresholding with mask
        
    @  FILES
        - this
        - objects-to-detect.jpg
        
    @  PROCEDURE
        1. Read RGB image.
        2. Converting to HSV.
        3. Getting Mask for blue objects.
        
    @  PS
        - The path
        to the image is relative to the root folder of the
        repository. You may
        need to change the path depending on your developing environment.
---------------------------------
'''

# Imports (our backstage crew)
import cv2
import os

# Setting the stage
images_path = "UY_03_Train_YOLO_for_Object_Detection/01_Class"

# ---------------------------------------------------------------------
# 1. CREATE THE SLIDE BARS TO TRACK THE COLORS.

# A function that does nothing, but with style.
def do_nothing(go_fish):
    print(f'The number is {go_fish}, and it shall remain so.')

# Lights, camera, window!
cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

# Cue the trackbars, our lovely supporting actors.
for color in ['blue', 'green', 'red']:
    for bound, init_val in [('min', 0), ('max', 255)]:
        cv2.createTrackbar(f'{bound}_{color}', 'Track Bars', init_val, 255, do_nothing)

# Execute it and have fun!
# while True:
#     if cv2.waitKey(0):
#         break

# ---------------------------------------------------------------------
# 2. READ AND PREPARE THE IMAGE.

# Read the image (like reading a script, but with more pixels)
image = f'{images_path}/objects-to-detect.jpg'
image_BGR = cv2.imread(image)

if image_BGR is not None:
    # Resize the image (our spotlight for today)
    image_BGR = cv2.resize(image_BGR, (600, 426))
else:
    print("The show can't go on. The image is missing! Check the image directory.")

# Showtime! Presenting: the original image.
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image_BGR)
cv2.waitKey(0)

# And now, for a colorful twist: converting BGR to HSV.
image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

# Ta-da! It's the HSV image.
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)
cv2.imshow('HSV Image', image_HSV)
cv2.waitKey(0)

# The show goes on as we define the loop for choosing the right colors for the mask.
while True:
    # Get the min and max values for each color from our trusty trackbars.
    # For minimum range
    min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
    min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
    min_red = cv2.getTrackbarPos('min_red', 'Track Bars')

    # For maximum range
    max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
    max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
    max_red = cv2.getTrackbarPos('max_red', 'Track Bars')

    # Implementing the mask with chosen colors from trackbars (our grand finale!).
    mask = cv2.inRange(image_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    # The mysterious masked image appears!
    cv2.namedWindow('Binary Image with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary Image with Mask', mask)

    # Press 'q' to gracefully exit this colorful performance.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the curtains and pack up.
cv2.destroyAllWindows()

print('\n####\nEncore! Encore! Well, that\'s it for now. Enjoy the rest of your day!')

# At this point, we have a window with the original image,
# another with the image converted to the HSV color system, and another with the mask.
# Additionally, we have a window with trackbars so you can adjust the mask values.
# You can indicate which pixels with red values should appear in the mask.
# The whiter the pixel, the more intense these values are.
# The darker the pixel, the less intense they are.
# We can use it to segmentation. Got it? Good!
