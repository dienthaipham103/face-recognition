import face_recognition
import cv2
import numpy as np

'''
This is a super simple (but slow) example of running face recognition on live video from your webcam.
There's a second example that's a little more complicated but runs faster.

PLEASE NOTE: This example requires OpenCV (the cv2 library) to be installed only to read from your webcam.
OpenCV is not required to use the face_recognition library. It's only required if you want to run this
specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
'''

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
person1_image = face_recognition.load_image_file('person1.jpd')
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

# Load a second sample picture and learn how to recognize it.
person2_image = face_recognition.load_image_file('person2.jpg')
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

# Load a second sample picture and learn how to recognize it.
person3_image = face_recognition.load_image_file('lac.jpg')
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding
]

known_face_names = [
    "Person1",
    "Person2",
    "Person3"
]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left, bottom), font, 0.3, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
