# import cv2 as cv
import cv2 as cv
import numpy as np
import time
from lib.Futils import Futils as fu

# ------------------------
# defines
coco_names      = fu.env('COCO_NAMES')
yolov4_cfg      = fu.env('YOLOV4_CFG')
yolov4_weights  = fu.env('YOLOV4_WEIGHTS')
gatinho         = fu.env('GATINHOS_JPEG_0')

# ------------------------
# Show (gatinho)
image_BGR = cv.imread(gatinho)
cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
cv.imshow('Original Image', image_BGR)
cv.waitKey(0)
# Get info about image
h, w = image_BGR.shape[:2]
print(f'Height: {h}, Width: {w}')

# ------------------------
# Load Yolo
network = cv.dnn.readNetFromDarknet(yolov4_cfg, yolov4_weights)
# Transform image into blob
blob = cv.dnn.blobFromImage(
    image_BGR,
    1 / 255.0,
    (416, 416),
    swapRB=True,
    crop=False
)
# Set blob as input to the network
network.setInput(blob)

# ------------------------
# Yolo was trained on the COCO dataset, which has 80 classes
# We need to load the class names from disk
# Get info about classes
with open(coco_names) as f:
    labels = [line.strip() for line in f]
# Get layers names
layers_names_all = network.getLayerNames()
print(layers_names_all)
# Get unconnected out layers
print("Unconnected out layers:", network.getUnconnectedOutLayers())
# Get unconnected out layers names
layers_names_output = [\
    layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
print("Unconnected out layers names:", layers_names_output)

# ------------------------
# Settings
# Set probability threshold
probability_minimum = 0.5
# Set threshold for filtering weak bounding boxes
threshold = 0.3
# Generate colors for bounding boxes
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# ------------------------
# Implementing forward pass
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()
print(f'Objects Detection took {(end - start):.5f} seconds')

# ------------------------
# Preparing lists for detected bounding boxes,
# obtained confidences and class's number
bounding_boxes = []
confidences = []
class_numbers = []

for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]
        
        if confidence_current > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w, h, w, h])
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)
# Suppression of non-maximums
results = cv.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
counter = 1

# ------------------------
# Drawing bounding boxes
# Checking if there is at least one detected object after non-maximum suppression
if len(results) > 0:
    # Going through indexes of results
    for i in results.flatten():
        # Showing labels of the detected objects
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

        # Incrementing counter
        counter += 1

        # Getting current bounding box coordinates,
        # its width and height
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        # Preparing colour for current bounding box
        # and converting from numpy array to list
        colour_box_current = colors[class_numbers[i]].tolist()

        # # # Check point
        # print(type(colour_box_current))  # <class 'list'>
        # print(colour_box_current)  # [172 , 10, 127]

        # Drawing boun1ding box on the original image
        cv.rectangle(image_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        # Preparing text with label and confidence for current bounding box
        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        # Putting text with label and confidence on the original image
        cv.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)
# ------------------------
# Comparing how many objects where before non-maximum suppression
# and left after

print()
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)

# Showing Original Image with Detected Objects
# Giving name to the window with Original Image
# And specifying that window is resizable
cv.namedWindow('Detections', cv.WINDOW_NORMAL)
# Pay attention! 'cv.imshow' takes images in BGR format
cv.imshow('Detections', image_BGR)
# Waiting for any key being pressed
cv.waitKey(0)
# Destroying opened window with name 'Detections'
cv.destroyWindow('Detections')