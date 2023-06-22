import face_recognition

ichi_image = face_recognition.load_image_file("1.jpg")
unknown_image = face_recognition.load_image_file("2.jpg")

ichi_encoding = face_recognition.face_encodings(ichi_image)[0]

for unknown_encoding in face_recognition.face_encodings(unknown_image):
    results = face_recognition.compare_faces([ichi_encoding], unknown_encoding)
    print(results)