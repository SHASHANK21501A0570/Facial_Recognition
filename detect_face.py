#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import os

########## KNN CODE ############
def distance(v1, v2):
    # Euclidean distance
    return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from the test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index], dk[index][0]

################################
def detect_faces():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

    dataset_path = "./face_dataset/"

    face_data = []

    labels = []
    class_id = 0
    names = {}

    # Dataset preparation
    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4]
            data_item = np.load(dataset_path + fx)
            face_data.append(data_item)
            target = class_id * np.ones((data_item.shape[0],))
            class_id += 1
            labels.append(target)

    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

    trainset = np.concatenate((face_dataset, face_labels), axis=1)



    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 50  # Adjust this threshold as needed

    while True:
        ret, frame = cap.read()
        if ret == False:
            continue
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect multi faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for face in faces:
            x, y, w, h = face
            # Get the face ROI
            offset = 5
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            face_section = cv2.resize(face_section, (100, 100))

            # Predict the label and distance
            out, dist = knn(trainset, face_section.flatten())
            print(dist)
            # Display name only if distance exceeds threshold
            if dist<5000:
                name = names[int(out)]
                
            else:
                name = 'Unknown'
                

            # Draw rectangle and display name
            cv2.putText(frame, name, (x, y-10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        cv2.imshow("Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

