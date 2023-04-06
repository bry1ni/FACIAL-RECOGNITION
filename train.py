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
names = list_of_files.copy()

faces_encodings = []
faces_names = []

# Training:
for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
    # Create array of known names
    names[i] = names[i].replace(cur_direc, "")
    # enlever les caracteres qu'on veut pas affichier
    x = names[i].replace(".jpeg", "")
    name = x.replace("/faces/", "")
    faces_names.append(name)
    np.save('faces_encodings.npy', faces_encodings)
    np.save('faces_names.npy', faces_names)
