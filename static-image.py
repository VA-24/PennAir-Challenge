import cv2

image = cv2.imread('shapes.png')

#experimented with a couple of blurring techniques and found that bilateralfilter worked the best
blurred = cv2.bilateralFilter(image, 30, 150, 150)

#edge detection
edges = cv2.Canny(blurred, 30, 170)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    #calculate center of contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        #draw center/outline
        cv2.drawContours(image, [contour], -1, (0, 0, 0), 3)
        cv2.circle(image, (cX, cY), 5, (0, 0, 0), -1)

cv2.imshow("Shapes Identified", image)
cv2.waitKey(0)
cv2.destroyAllWindows()