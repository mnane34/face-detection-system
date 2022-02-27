import cv2
import numpy as np

video = cv2.VideoCapture("football.mp4")
faceCascade = cv2.CascadeClassifier("frontalface.xml")

while True:
    _,frame = video.read()
    frame = cv2.resize(frame,(840,600))
    imageGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imageGray,1.06,20)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(frame,'Face Detected!',(x-20,y-20), font, 0.4, (0,255,0), 1, cv2.LINE_AA)
    
    cv2.imshow("Frame",frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
video.release()
cv2.destroyAllWindows()



