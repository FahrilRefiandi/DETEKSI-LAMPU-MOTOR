import cv2 as cv

cap = cv.VideoCapture('1.mp4')


a = 0
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

while True:
    a += 1
    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(height,width)
    roi = frame[200:352,300:640]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

    mask = object_detector.apply(roi)
    contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv.boundingRect(cnt)
            cv.imwrite('dataset/'+str(a)+'.jpg',gray[0:200,0:200])
            cv.rectangle(roi,(x,y),(x+w , y+h),(0,255,0),3)
            # hitung jumlah motor
    cv.imshow("video",frame)

    if cv.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()