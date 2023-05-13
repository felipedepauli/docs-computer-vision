# imports
import cv2 as cv
import numpy as np
import time

# definitions
writer = None
h, w, = None, None

# load yolo labels
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

print('List with labels names:')
print(labels)

# load yolo network
network = cv.dnn.readNetFromDarknet("./yolo-coco-data/yolov3.cfg", "./yolo-coco-data/yolov3.weights")

# create a list with all the output layers names
layers_names_all = network.getLayerNames()

layers_names_output = [\
    layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

# setting minimum probability to eliminate weak predictions
probability_minimum = 0.5
# setting threshold for non-maximum suppression
threshold = 0.3

# generate colors for representing every detected object
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

f = 0 # Counting frames
t = 0 # Counting total time

video = cv.VideoCapture("./videos/traffic-cars.mp4")
while True:
    ret, frame = video.read()
    
    if not ret:
        break;
    
    if w is None or h is None:
        h, w = frame.shape[:2]
     
    # the image has to be converted to a blob    
    blob = cv.dnn.blobFromImage(
        frame,      # input image
        1/255.0,    # normalization factor
        (416, 416), # size (416, 416) is recommended for yolo
        swapRB=True,# swap red and blue channels
        crop=False  # no cropping
    )
    
    network.setInput(blob) # set the blob as input to the network
    start = time.time()    # start the timer
    # You can get all outputs from the network by calling forward() method
    # and passing names of the layers you want to get outputs from
    # In this case, we will get the names of the output layers
    output_from_network = network.forward(layers_names_output) # forward pass
    end = time.time()      # stop the timer
    
    f += 1
    f += end - start
    
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    bounding_boxes = []
    confidences = []
    class_numbers = []
    
#     the YOLO model has multiple output layers because it detects objects at multiple scales. The output from each of these layers includes information on bounding boxes, objectness score, and class probabilities for multiple grid cells of a particular scale.
# Specifically, for YOLOv3, there are three output layers corresponding to three different scales. For each scale, the model divides the input image into a grid (e.g., 13x13, 26x26, 52x52), and each grid cell predicts a fixed number of bounding boxes. For each bounding box, the model predicts coordinates (x, y, width, height), an objectness score, and class probabilities for all classes (e.g., 80 classes in the COCO dataset).
# After the forward pass through the network, the output includes the predictions from all of these layers. The script processes these outputs to extract the class with the highest probability for each bounding box and uses this information to draw bounding boxes and labels on the video frames.
    
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
    
# Suppression of non-maximum boxes
    results = cv.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            
            color_box_current = colors[class_numbers[i]].tolist()
            
            cv.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), color_box_current, 2)
            
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
            
            cv.putText(frame, text_box_current, (x_min, y_min - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color_box_current, 2)
            
                # Initializing writer
        # we do it only once from the very beginning
        # when we get spatial dimensions of the frames
    if writer is None:
        # Constructing code of the codec
        # to be used in the function VideoWriter
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        # Pay attention! If you're using Windows, yours path might looks like:
        # r'videos\result-traffic-cars.mp4'
        # or:
        # 'videos\\result-traffic-cars.mp4'
        writer = cv.VideoWriter('videos/result-traffic-cars.mp4', fourcc, 30,
                                (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)
    
    # Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer
video.release()
writer.release()