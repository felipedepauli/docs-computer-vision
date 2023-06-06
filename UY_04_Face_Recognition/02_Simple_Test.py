import cv2
import face_recognition

# Load the image to detect
image_to_detect = cv2.imread('media/Copa2006.jpg')

# Detect all faces in the image
# The second argument of face_locations can be 'cnn' or 'hog'
all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')

# Print the number of faces detected and show the original image
print('There are {} people in this image'.format(len(all_face_locations)))
cv2.imshow("Oiew", image_to_detect)

# Loop through each face in this image
for index, current_face_location in enumerate(all_face_locations):
    # Splitting the tuple to get the four position values of current face
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at top: {}, right: {}, bottom: {}, left: {}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))

    # Slice image array by positions inside the loop
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    
    # Show each face in a window
    cv2.imshow("Face No " + str(index+1), current_face_image)
    cv2.waitKey(0)
cv2.DestroyAllWindows()