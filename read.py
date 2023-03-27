import cv2 as cv
from tracker import *

cap = cv.VideoCapture("1.mp4")
a = 0
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

tracker = EuclideanDistTracker()

while True:
    a += 1
    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(height,width)
    roi = frame[200:352,300:640]
    roi2 = frame[0:200,450:530]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    
    mask = object_detector.apply(roi)
    contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
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

    #cv.imshow("mask",mask)
    #cv.imshow("awal",gray)   
    #cv.imshow("roi",roi)
    cv.imshow("roi2",roi2)
    cv.imshow("video",frame)

    if cv.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()