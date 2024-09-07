import cv2
import numpy as np

cap = cv2.VideoCapture('shapes.mp4')

while True:

    ret, frame = cap.read()

    blurred = cv2.GaussianBlur(frame, (19, 19), 0)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opened, 30, 170)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        #calculate center of countour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            #draw center/outline
            cv2.drawContours(frame, [contour], -1, (0, 0, 0), 3)
            cv2.circle(frame, (cX, cY), 5, (0, 0, 0), -1)

    cv2.imshow('Shapes Identified', frame)

    #use q key to break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()