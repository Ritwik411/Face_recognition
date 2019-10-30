# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:46:18 2019

@author: Ritwik Gupta
"""

import face_recognition
from sklearn import svm
import os
import cv2

encodings = []
names = []
path = 'E:\Study Material\Machine Learning\Face_recognition\images'
test_path = 'E:\Study Material\Machine Learning\Face_recognition'
num = 0
    # Importing html smaples for Face and Eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Starting your Laptop Camera for recording
id1 = input('Enter your name')
cap = cv2.VideoCapture(0)
while num <2:
    ret, img = cap.read()  # Getting image from the camera
        #ret would be 1 if python can read data else 0      
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Converting Image to GrayScale
        
        # Detecting Faces from image using Face_Samples
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
        # Iterating for the dimentions of the Detected faces to draw rectangle
    for (x,y,w,h) in faces:
            
            # Writing the image to the specified location
        cv2.imwrite(test_path+ "\\test.jpg", gray[y:y+h, x:x+w])    
            
            # Creating Rectangle around Face with color Red and of width 2
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('img',img) # Displays the Image with rectangles on Face
        num+=1
cap.release()
cv2.destroyAllWindows()

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
# Training directory
train_dir = os.listdir(path)

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir(path + "\\"+person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(path +"\\"+person + "\\" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains none or more than faces, print an error message and exit
        if len(face_bounding_boxes) != 1:
            print(person + "/" + person_img + " is improper and can't be used for training.")
            #exit()
        else:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('test.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    if name[0] == str.lower(id1).strip():
        print("Congratulations, "+name[0]+" you are authenticated!")
    else :
        print("Sorry, you're not authenticated!")

import pickle
with open('model','wb') as f:
    pickle.dump(clf,f)

    