import cv2
import face_recognition

# Load sample images and extract face encodings
modi_image = face_recognition.load_image_file("media/Modi.jpg")
modi_face_encoding = face_recognition.face_encodings(modi_image)[0]
# ----
trump_image = face_recognition.load_image_file("media/Trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [modi_face_encoding, trump_face_encoding]
known_face_names = ["Narendra Modi", "Donald Trump"]

# Load an unknown image and find all the faces and face encodings in it
image_to_detect = face_recognition.load_image_file("media/Modi_Trump.jpeg")
original_image = cv2.imread("media/Modi_Trump.jpeg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(image_to_detect, model="hog")
all_face_encodings = face_recognition.face_encodings(image_to_detect, face_locations)

print('There are {} no of faces in this image'.format(len(face_locations)))

# Loop through each face found in the unknown image
for current_face_location, current_face_encoding in zip(face_locations, all_face_encodings):
    
    # Splitting the tuple to get the four position values of current face
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    
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
    cv2.rectangle(original_image, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)

    # 12. Write name below each face
    font = cv2.FONT_HERSHEY_DUPLEX
    if (name == "Unknown Face"):
        continue
    cv2.putText(original_image, name, (left_pos, bottom_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
# 13. Show the image with rectangle and text
cv2.imshow("Faces Identified", original_image)
cv2.waitKey(0)




