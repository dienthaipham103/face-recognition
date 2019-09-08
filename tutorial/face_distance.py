import face_recognition

known_image = face_recognition.load_image_file('biden.jpg')
unknown_image = face_recognition.load_image_file('obama.jpg')

known_image_encoding = face_recognition.face_encodings(known_image)[0]
unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]

distance = face_recognition.face_distance([known_image_encoding], unknown_image_encoding)
print(distance)

