#import library
import cv2
from tracker import *


tracker=EuclideanDistTracker() #import EuclideanDistTracke class from the tracker.py
cap=cv2.VideoCapture("istockphoto-472833371-640_adpp_is.mp4")

object_detector=cv2.createBackgroundSubtractorMOG2(history=170,varThreshold=55)


while True:
    ret ,frame =cap.read()
    height ,weidth ,_ =frame.shape
    
    roi=frame[100:320,10:630]
    
    mask=object_detector.apply(roi)
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    #cv2.line(frame,(25,200),(1200,200),(255,127,0),3)
    
    counters,_ =cv2.findContours(mask, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    detections=[]
    for cnt in counters:
        area =cv2.contourArea(cnt)
        if area > 350:
            #cv2.drawContours(frame,[cnt],-1,(0,255,0),3)
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])
    
    
    #object tracking
    boxes_ids=tracker.update(detections)
    
    
    #count number of cars
    for box_id in boxes_ids:
        x,y,w,h,id =box_id
        cv2.putText(roi,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)

    
    
    cv2.imshow("roi",roi)
    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)
    
    key=cv2.waitKey(30)
    if key == 27:
        break
        

cap.release()
cv2.destroyAllWindows()