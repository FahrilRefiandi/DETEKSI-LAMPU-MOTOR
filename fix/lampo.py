import cv2 as cv

img = cv.imread('./datasetfix/12.jpg')
# dataset no 12 dan 30,31 dan 58 aman
cv.imshow('img', img)   

rszImg = cv.resize(img, (int((img.shape[1] * 3)), int(img.shape[0] * 3))) 
cv.imshow('rszImg', rszImg)

gray = cv.cvtColor(rszImg, cv.COLOR_BGR2GRAY)
threshold_value = 127

gausian = cv.GaussianBlur(gray,(11,11),0)
median = cv.medianBlur(gausian, 9)
cv.imshow("median", median)
ret, mask = cv.threshold(median, threshold_value, 255, cv.THRESH_BINARY)
cv.imshow("mask", mask)

contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
a = 0
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    area = cv.contourArea(contour)
    a+=1

    # untuk supra
    if area > 100 and area < 500 :
        if w > h :
            print(f"x:{x} y:{y} w:{w} h:{h}")
            cv.rectangle(rszImg,(x,y),(x+w , y+h),(0,255,0),3)
            cv.imwrite('roi-lampu/'+str(a)+'.jpg', gray[y:y+h,x:x+w])
    # untuk scoopy
    elif area > 200 and area < 2000 :
        if w < h :
            print(f"x:{x} y:{y} w:{w} h:{h}")
            cv.rectangle(rszImg,(x,y),(x+w , y+h),(0,255,0),3)
            cv.imwrite('roi-lampu/'+str(a)+'.jpg', gray[y:y+h,x:x+w])
    if area > 500 and area < 1000 :
        if w/h >= 1 :
            cv.rectangle(rszImg,(x,y),(x+w , y+h),(0,255,0),3)
            cv.imwrite('roi-lampu/'+str(a)+'.jpg', gray[y:y+h,x:x+w])

    print(f"x:{x} y:{y} w:{w} h:{h}")
    print(area)
        


cv.imshow("result", rszImg)

# menunggu tombol keyboard ditekan
cv.waitKey(0)

# menutup semua jendela yang terbuka
cv.destroyAllWindows()