import cv2 as cv

cap = cv.VideoCapture('video-khuluq.mp4')

a = 0
object_detector = cv.createBackgroundSubtractorKNN(history=70, dist2Threshold=200,detectShadows=False)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    #print(height,width)
    roi = frame[200:352,300:640]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    median = cv.medianBlur(gray, 7)
    gousian = cv.GaussianBlur(median,(41,41),0)
    
    mask = object_detector.apply(gousian)

    
    contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:    
        area = cv.contourArea(cnt)
        if area > 3000 and area < 6000:
            x, y, w, h = cv.boundingRect(cnt)
            # if h > 90 and w < 80 and w > 60:
            if h/w > 1.45:
                a += 1
                cv.rectangle(roi,(x,y),(x+w , y+h),(0,255,0),3)
                cv.imwrite('dataset-non-gray/'+str(a)+'.jpg',roi[y:y+h,x:x+w])
                print(f"x:{x} y:{y} w:{w} h:{h}")
            # hitung jumlah motor
    cv.imshow("video",frame)
    cv.imshow("mask",mask)
    
    if cv.waitKey(2) & 0xff == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()