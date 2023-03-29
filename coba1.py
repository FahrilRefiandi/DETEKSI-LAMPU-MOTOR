import cv2 as cv
from tracker import *
import numpy as np

cap = cv.VideoCapture("1.mp4", cv.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.uint8)

object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=60,detectShadows=False)
tracker = EuclideanDistTracker()
 
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    #print(height,width)
    roi = frame[200:352,300:640]
    roi2 = frame[50:200,430:530]

    erosion = cv.erode(frame,kernel,iterations = 1)

    #gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    
    opening = cv.morphologyEx(roi, cv.MORPH_OPEN, kernel)

    mask = object_detector.apply(erosion)
    contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        #print(area)
        if area > 1000:
            print(area)
            x, y, w, h = cv.boundingRect(cnt)
            #cv.imwrite('dataset/'+str(a)+'.jpg',gray[y:y+h,x:x+w])
            #cv.rectangle(roi,(x,y),(x+w , y+h),(0,255,0),3)

            detections.append([x, y, w, h])
    
        # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("mask",mask)
    #cv.imshow("awal",gray)   
    #cv.imshow("roi",roi)
    #cv.imshow("roi2",roi2)
    cv.imshow("video",frame)
    #cv.imshow("erosi",erosion)

    if cv.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()