import face_recognition
import cv2
import numpy as np
import os
import glob

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cur_direc = os.getcwd()
path = os.path.join(cur_direc, 'faces/')
list_of_files = [f for f in glob.glob(path + '*.jpeg')]
number_files = len(list_of_files)
names = []
for file in list_of_files:
    name = file.replace(cur_direc, "").replace(".jpeg", "").replace("/faces/", "")
    names.append(name)

faces_encodings = np.load('faces_encodings.npy')
faces_names = np.load('faces_names.npy')


if len(faces_names) == len(names):
    pass
else:
    # case: new faces on the faces/ folder
    # we only encode the new faces and append the results
    new_names = list(set(names) - set(list(faces_names)))
    for name in new_names:
        print(name)
        image = face_recognition.load_image_file(os.path.join(path, name + '.jpeg'))
        image_encoding = face_recognition.face_encodings(image)[0]
        faces_encodings = np.append(faces_encodings, image_encoding)
        faces_names = np.append(faces_names, name)


face_locations = []
face_encodings = []
face_names = []


def faceRecognetion(frame):
    process_this_frame = True
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # we detect and encode faces on the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # we compare between results
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"
            # euclidienne distance
            face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            #
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        font = cv2.FONT_HERSHEY_DUPLEX

        if name != 'Unknown':
            # Draw a green rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Input text label with a name, acessType below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:  # if the face doesnt exist in our data base
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    faceRecognetion(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
