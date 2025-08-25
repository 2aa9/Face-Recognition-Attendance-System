
### `attendance.py`
```python
import face_recognition
import cv2
import os
from datetime import datetime

# Load known faces
known_faces = []
known_names = []
for file in os.listdir("known_faces"):
    img = face_recognition.load_image_file(f"known_faces/{file}")
    encoding = face_recognition.face_encodings(img)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(file)[0])

# Initialize webcam
cap = cv2.VideoCapture(0)
attendance = []

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for face_encoding, face_location in zip(encodings, faces):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
            if name not in attendance:
                attendance.append(name)
                print(f"{datetime.now()}: {name} marked present")

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
