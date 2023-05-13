'''
---------------------------------
    @  COURSE NAME
        Course:  Training YOLO v3 for Objects Detection with Custom Data

    @  SECTION
        Quick Win - Step 2: Simple Object Detection by thresholding with mask

    @  FILES
        - this
        - camera's frames

    @  PROCEDURE
        1. Set the threshold colors.
        2. Read RGB image from camera.
        3. Convert image to HSV color space.
        4. Apply color mask to isolate the desired object.
        5. Find contours of the masked object.
        6. Extract bounding rectangle coordinates.
        7. Draw a bounding box around the detected object.
        8. Add a label to the detected object.

---------------------------------
'''



# Who's ready for some object detection magic? Let's get started!
#
# Target:
# Window with Detected Object, Bounding Box and Label in Real Time


# Importing the magical libraries that makes all of this possible
import cv2
import numpy as np

# 1. SET THE THRESHOLD COLORS.

# Defining lower bounds and upper bounds of the mask for our mysterious object
color_bounds = {
    "blue": {
        "lower": np.array([100, 150, 0]),
        "upper": np.array([140, 255, 255])
    },
    # The explanation of why this is this way at the bottom of the file
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
color_name = "pink"

# Time to find out which version of OpenCV we're using!
# Split the version string by dots and grab the first part
v = cv2.__version__.split('.')[0]

# 2. READ RGB IMAGE FROM CAMERA.

# Let's create an object for reading video from the camera
camera = cv2.VideoCapture(0)
# Ps. you cannot be in wsl to use the camera

# Loop time! Catching frames and making magic happen
while True:
    # Capture frame-by-frame from the camera
    _, frame_BGR = camera.read()

    # 3. CONVERT IMAGE TO HSV COLOR SPACE.

    frame_HSV = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV)

    # 4. APPLY COLOR MASK TO ISOLATE THE DESIRED OBJECT.
    
    # Create the mask for the chosen color
    if color_name == "red":
        mask1 = cv2.inRange(frame_HSV, color_bounds[color_name]["lower"], color_bounds[color_name]["upper"])
        mask2 = cv2.inRange(frame_HSV, color_bounds[color_name]["lower2"], color_bounds[color_name]["upper2"])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(frame_HSV, color_bounds[color_name]["lower"], color_bounds[color_name]["upper"])

    # Show the frame with the mask applied
    cv2.namedWindow('Binary frame with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary frame with Mask', mask)

    # 5. FIND CONTOURS OF THE MASKED OBJECT.
    # the outline of our object
    
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
        # cv2.drawContours(frame_BGR, contours, 0, (0, 255, 0), 3)


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


# A conversão da imagem para o espaço de cores HSV (Hue, Saturation, Value) é feita
# porque é mais fácil e eficiente lidar com a detecção de cores no espaço HSV do que no espaço RGB (Red, Green, Blue).

# No espaço de cores HSV:
# - Hue (Matiz): Representa a cor, variando de 0 a 180 (na maioria das implementações do OpenCV) ou de 0 a 360 (em outros sistemas).
# - Saturation (Saturação): Representa a quantidade de cor (pureza), variando de 0 (sem cor) a 255 (cor pura).
# - Value (Valor): Representa a luminosidade, variando de 0 (preto) a 255 (branco).

# As principais vantagens de usar o espaço de cores HSV para detecção de cores são:

# - A separação da cor (matiz) da luminosidade e saturação: No espaço HSV, a cor é representada apenas pelo componente Hue,
#   tornando mais fácil isolar uma cor específica em diferentes condições de iluminação.

# - Maior robustez a variações de iluminação: O espaço de cores HSV é menos sensível a variações de iluminação do que o espaço RGB,
#   o que é importante na detecção de cores, já que a iluminação pode variar significativamente em diferentes cenários.

# - Menor complexidade computacional: Como a cor é representada apenas pelo componente Hue,
#   você pode criar máscaras para detecção de cores usando apenas um intervalo de valores Hue,
#   reduzindo a complexidade computacional em comparação com a detecção de cores no espaço RGB,
#   que exigiria intervalos para os três componentes (R, G e B).

# - Essas características tornam o espaço de cores HSV uma escolha popular e eficiente para detecção de cores
#   em aplicações de processamento de imagem e visão computacional.

# ------------------------------ RED COLOR --------------------------------
# The red color has a peculiarity in the HSV color space that sets it apart from other colors. Red is found at the beginning and end of the hue spectrum, with values close to 0 and close to 180. This causes the hue value range for red to be split into two parts.

# For this reason, the code defines two masks for the red color:

# mask1 is created for the lower hue value range for red (0 to some lower value, in the given example, 10).
# mask2 is created for the upper hue value range for red (some higher value, in the given example, 160, to 180).
# Then, the two masks are combined using the bitwise OR operation (cv2.bitwise_or). This means that pixels that match either of the masks (i.e., are in the lower or upper red hue ranges) will be included in the final mask.

# For other colors, such as green, blue, yellow, and pink, the hue ranges are continuous and not split into two parts, so just one mask is sufficient to capture the color.