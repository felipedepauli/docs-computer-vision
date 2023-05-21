import cv2
import face_recognition

# Capture video from webcam
video_capture = cv2.VideoCapture(0)
cv2.namedWindow('AMORINHOS', cv2.WINDOW_NORMAL)


# Initialize variables
all_face_locations  = []
all_face_encodings  = []
all_face_names      = []

# ------------------------------
# Load sample images and extract face encodings
modi_image = face_recognition.load_image_file("../media/Modi.jpg")
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]
# ----
trump_image = face_recognition.load_image_file("../media/Trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]
# ----
felps_image = face_recognition.load_image_file("../media/Felps.jpeg")
felps_image_encoding = face_recognition.face_encodings(felps_image)[0]
# ----
amorinha_image = face_recognition.load_image_file("../media/Amorinha.jpeg")
amorinha_image_encoding = face_recognition.face_encodings(amorinha_image)[0]
# ------------------------------

# Create arrays of known face encodings and their names
known_face_encodings = [modi_face_encoding, trump_face_encoding, felps_image_encoding, amorinha_image_encoding]
known_face_names = ["Narendra Modi", "Donald Trump", "Felps", "Lindinha"]

print('There are {} no of faces in this image'.format(len(all_face_locations)))

# Create an outer while loop to loop over frames
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    # This is optional
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Detect all faces in the image
    # The second argument of all_face_locations can be 'cnn' or 'hog'
    all_face_locations = face_recognition.face_locations(small_frame, model='HOG')
    all_face_encodings = face_recognition.face_encodings(small_frame, all_face_locations)

    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        
        # Splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        top_pos     *= 4
        right_pos   *= 4
        bottom_pos  *= 4
        left_pos    *= 4
        
        # Compare faces and get the matches list (inside the loop)
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)

        # Initialize a name string as unknown face
        name = "Unknown Face"
        
        # User first match and get name from the respective index
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name = known_face_names[first_match_index]
            
        if (name == "Unknown Face"):
            continue
        
        # Draw rectangle around the face detected
        cv2.rectangle(frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)

        # 12. Write name below each face
        font = cv2.FONT_HERSHEY_DUPLEX
        if (name == "Unknown Face"):
            continue
        cv2.putText(frame, name, (left_pos, bottom_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        

    
    # 13. Show the image with rectangle and text
    cv2.imshow("AMORINHOS", frame)
    
        # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        break

