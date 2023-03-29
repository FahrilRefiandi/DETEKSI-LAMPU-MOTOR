import cv2 as cv
from tracker import *
import numpy as np

kernel = np.ones((5,5),np.uint8)
cap = cv.VideoCapture("khuluq/1.mp4", cv.IMREAD_GRAYSCALE)
a = 0
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=60,detectShadows=False)

tracker = EuclideanDistTracker()

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    #print(height,width)
    roi = frame[200:352,300:640]
    roi2 = frame[50:200,430:530]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    
    mask = object_detector.apply(roi)
    erosion = cv.erode(mask,kernel,iterations = 1)
    dilation = cv.erode(erosion,kernel,iterations = 8)


    contours, hie = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        # print(hie)
        #print(area)
        if area > 800 and area < 1500:
            # print(area)
            a += 1
            x, y, w, h = cv.boundingRect(cnt)
 
            # print("y" + y)
            # print("h" + h)
            
            cv.imwrite('khuluq/ds/'+str(a)+'.jpg',gray[0:200, 0:200])
            cv.rectangle(frame,(x,y),(x+w , y+h),(0,255,0),3)
            cv.rectangle(roi,(x,y),(x+w , y+h),(0,255,0),3)

            detections.append([x, y, w, h])
    
        # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        print("ini id : " , id)
        print(x)
        print(w)
        cv.putText(roi,str(area), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("roi", roi)
    cv.imshow("frame", frame)
    cv.imshow("mask",mask)
    cv.imshow("erosion", erosion)
    cv.imshow("dilation", erosion)

    if cv.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()