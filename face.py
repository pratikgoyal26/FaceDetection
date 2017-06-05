import cv2
import numpy as np
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
count=0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        face=img[x,y]
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,232,0),2)
        cv2.imwrite("C:/Users/pratik/Desktop/Face detection/Image/face%d.jpg" % count,roi_gray) 
    cv2.imshow('Image',img)
    
    k= cv2.waitKey(30) & 0xff
    if k == 27:
        break
    count += 1
cap.release()
cv2.destroyAllWindows()
    
