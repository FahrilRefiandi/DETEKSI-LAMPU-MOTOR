import cv2 as cv

cap = cv.VideoCapture("1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[200:352,300:640]
    roi2 = frame[0:200,450:530]
    
    # Resize the input matrices to have the same dimensions
    roi_resized = cv.resize(roi, (roi2.shape[1], roi2.shape[0]))
    
    # Combine the input matrices horizontally
    combined_frame = cv.hconcat([roi_resized, roi2])

    # Display the combined frame
    cv.imshow('Combined Frame', combined_frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release the video object and close any open windows
cap.release()
cv.destroyAllWindows()
