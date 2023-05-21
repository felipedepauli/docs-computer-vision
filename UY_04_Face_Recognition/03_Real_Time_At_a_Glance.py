import face_recognition
import cv2

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_locations = []

# Create an outer while loop to loop over frames
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    # This is optional
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Detect all faces in the image
    # The second argument of face_locations can be 'cnn' or 'hog'
    face_locations = face_recognition.face_locations(frame, model='HOG')

    # Loop through each face in this frame of video
    for current_face_location in face_locations:
        # Splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        print('Found face at top: {}, right: {}, bottom: {}, left: {}'.format(top_pos, right_pos, bottom_pos, left_pos))
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
    cv2.imshow('Video', frame)


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        break

