import numpy as np
import cv2 #importing modules




face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #initializing face cascades

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0) #variable for video capture

while 1:
    ret, img = cap.read() #taking return value from video reading
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #converting into gray image
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) #tracking face using faceCascade
    
    for (x,y,w,h) in faces:         #drawing recangles
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray) #recangles for eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
            
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
