import cv2

# Read the image
image = cv2.imread('../fix/dataset-primer/2.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to convert the image to binary
ret, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Apply morphological operations to remove noise and fill holes
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opening = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

# Subtract the background
background = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, foreground = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Foreground Image', foreground.astype('uint8'))

# Wait for user input
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()
