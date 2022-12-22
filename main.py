import cv2 as cv
from tracker import *

#Enter your Video
vid=cv.VideoCapture("highway.mp4")

#tracker object
tracker=EuclideanDistTracker()


#Object detection
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=100)


while True:
    ret,frame=vid.read() #reading frames
    height,width,_=frame.shape
    #region of intrest where detection will occur
    roi=frame
    #roi=frame[340:720,100:1000]

    #1.Object Detection
    mask=object_detector.apply(roi)#apply mask to turn image greyscale
    _,mask=cv.threshold(mask,254,255,cv.THRESH_BINARY)#Remove shadows(only 254to255 allowed)
    contours , _ =cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    detections=[]   #array to store rectageles of objects
    for cnt in contours:
        area=cv.contourArea(cnt)
        if area>100:
            #for making rectangles on the objects
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x,y,w,h])  #saving dimensions of objects


    #2.Object Tracking
    boxes_id=tracker.update(detections)
    for box_id in boxes_id:
        x,y,w,h,id = box_id
        cv.putText(roi,str(id),(x,y-15),cv.FONT_HERSHEY_PLAIN,1,(155,0,0),2)
        cv.rectangle(roi, (x,y), (x+w, y+h), (0,255,0),3)


    cv.putText(frame, 'Object Detection',(50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8, cv.LINE_4)
    cv.putText(frame, 'Press X to exit',(60, 80), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv.imshow("frame",frame)
    #cv.imshow("Maks",mask)
    key=cv.waitKey(30)
    if key in [ord('x'), 1048673]:
        vid.release()
        cv.destroyAllWindows()
    elif key==27:
        break


vid.release()
cv.destroyAllWindows()
