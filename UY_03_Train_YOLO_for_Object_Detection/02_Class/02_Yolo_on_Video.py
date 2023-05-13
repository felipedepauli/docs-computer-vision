

# 1. Reading frames from a video in loop
# 2. Getting blob from the frame
# 3. Implementing Forward Pass
# 4. Getting Bounding Box co-ordinates, confidence score, class-id from the output
# 5. Non-maximum Suppression
# 6. Drawing bounding box with class label on the image
# 7. Writing processed image back to a video

# Result: New video file with detected objects, boungin boxes and labels

# Imports

import cv2
import numpy as np
import time

video = cv2.VideoCapture("./videos/traffic-cars.mp4")
writer = None
# Preparing variables for spatial dimensions of the frames
h, w = None, None

# Loading Yolo v3 network

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# Check point
print('List with labels names:')
print(labels)

network = cv2.dnn.readNetFromDarknet("./yolo-coco-data/yolov3.cfg", "./yolo-coco-data/yolov3.weights")

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

layers_names_output = [\
    layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5
# Setting minimum threshold for non-maximum suppression
threshold = 0.3

# Generating randomly colours for representing every detected object
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

f = 0   # Definig variable for counting frames
t = 0   # Defining variable for counting total time

# Defining loop for catching frames
while True:
    # Retrieved and frame from the video
    ret, frame = video.read()   # Reading frames
    
    if not ret:
        break #
    
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]
        
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )
    
    network.setInput(blob)  # Implementing forward pass
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    
    f+=1
    f+= end - start
    
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
    
    
    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes  = []
    confidences     = []
    class_numbers   = []
    
    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # Check point
            # Every 'detected_objects' numpy array has first 4 numbers with
            # bounding box coordinates and rest 80 with probabilities for every class

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial image size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original image and in this way get coordinates for center
                # of bounding box, its width and height for original image
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
             
            # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence

    # It is needed to make sure that data type of the boxes is 'int'
    # and data type of the confidences is 'float'
    # https://github.com/opencv/opencv/issues/12789
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """

    # Checking if there is at least one detected object
    # after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
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

            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    """
    End of:
    Drawing bounding boxes and labels
    """

    """
    Start of:
    Writing processed frame into the file
    """

    # Initializing writer
    # we do it only once from the very beginning
    # when we get spatial dimensions of the frames
    if writer is None:
        # Constructing code of the codec
        # to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        # Pay attention! If you're using Windows, yours path might looks like:
        # r'videos\result-traffic-cars.mp4'
        # or:
        # 'videos\\result-traffic-cars.mp4'
        writer = cv2.VideoWriter('videos/result-traffic-cars.mp4', fourcc, 30,
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


"""
Some comments

What is a FOURCC?
    FOURCC is short for "four character code" - an identifier for a video codec,
    compression format, colour or pixel format used in media files.
    http://www.fourcc.org


Parameters for cv2.VideoWriter():
    filename - Name of the output video file.
    fourcc - 4-character code of codec used to compress the frames.
    fps	- Frame rate of the created video.
    frameSize - Size of the video frames.
    isColor	- If it True, the encoder will expect and encode colour frames.
"""
