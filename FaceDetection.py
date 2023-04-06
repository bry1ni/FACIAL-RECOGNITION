import os
import cv2

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


def faceDetection(frame):
    cropped_faces = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cropped_face = frame[y+2:y + h-2, x+2:x + w-2]
        cropped_faces.append(cropped_face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return frame, cropped_faces


image = cv2.imread('/Users/rayanpicso/Desktop/MyProjects/Face recognition/MultipleFaces/iftar2022HIS.jpeg')
result, faces = faceDetection(image)
cv2.imshow('result', result)
for i, face in enumerate(faces):
    cv2.imshow(f'face{i}', face)

cv2.waitKey(0)
