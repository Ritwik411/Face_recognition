# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:56:29 2019

@author: Ritwik Gupta
"""

from flask import Flask,request,jsonify,render_template
import pickle
import face_recognition
import cv2
import os
from sklearn import svm

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/Login.html')
def login():
    return render_template('Login.html')

@app.route('/verify.html')
def verify():
    return render_template('verify.html')

@app.route('/predict',methods=['POST'])
def predict():
    id1 = request.form['un']
    num = 0
    cap = cv2.VideoCapture(0)
    while num <2:
        ret, img = cap.read()  # Getting image from the camera
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Converting Image to GrayScale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.imwrite("img\\test.jpg", gray[y:y+h, x:x+w])             
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('img',img) 
            num+=1
    cap.release()
    cv2.destroyAllWindows()
    test_image = face_recognition.load_image_file("img\\test.jpg")
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[i]
        person = model.predict([test_image_enc])
    if id1==person[0]:
        text = 1
    else:
        text = 0
    return render_template('verify.html',prediction_text = text)

@app.route('/sign_up.html')
def sign_up():
    return render_template('sign_up.html')

@app.route('/database',methods=['POST'])
def database():
    new_id = request.form['username'].lower()
    flag = 1
    try:
        os.mkdir("img\\"+new_id)
        print("Directory " , "img\\"+ new_id ,  " Created ") 
        num = 0
        cap = cv2.VideoCapture(0)
        while num <20:
            ret, img = cap.read()  # Getting image from the camera
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Converting Image to GrayScale
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.imwrite("img\\"+str(new_id)+ "\\" +str(num)+ ".jpg", gray[y:y+h, x:x+w])    
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.imshow('img',img) # Displays the Image with rectangles on Face
                num+=1
        cap.release()
        cv2.destroyAllWindows()
        train_dir = os.listdir("img")
        encodings = []
        names = []
        for person in train_dir:
            pix = os.listdir("img\\" + person)
            for person_img in pix:
                face = face_recognition.load_image_file("img" +"\\"+person + "\\" + person_img)
                face_bounding_boxes = face_recognition.face_locations(face)                
                if len(face_bounding_boxes) != 1:
                    print(person + "/" + person_img + " is improper and can't be used for training.")
                else:
                    face_enc = face_recognition.face_encodings(face)[0]
                encodings.append(face_enc)
                names.append(person)
        clf = svm.SVC(gamma='scale')
        clf.fit(encodings,names)
        flag=1
        import pickle
        with open('model.pkl','wb') as f:
            pickle.dump(clf,f)
    except FileExistsError:
        flag=0
    return render_template('Login.html',prediction_text = flag) 
    
        

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/gallery.html')
def gallery():
    return render_template('gallery.html')

@app.route('/package.html')
def package():
    return render_template('package.html')

@app.route('/goa.html')
def goa():
    return render_template('goa.html')

@app.route('/Manali.html')
def Manali():
    return render_template('manali.html')

@app.route('/ooty.html')
def ooty():
    return render_template('ooty.html')

@app.route('/darjeeling.html')
def darjeeling():
    return render_template('darjeeling.html')

@app.route('/cart1.html')
def cart1():
    return render_template('cart1.html')

@app.route('/cart2.html')
def cart2():
    return render_template('cart2.html')

@app.route('/cart3.html')
def cart3():
    return render_template('cart3.html')

@app.route('/cart4.html')
def cart4():
    return render_template('cart4.html')

if __name__ == "__main__":
    app.run(debug=True)

