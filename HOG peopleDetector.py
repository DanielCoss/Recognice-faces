import cv2 as cv
import imutils

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture(0)

while True:
    x, image = cap.read()
    image = imutils.resize(image, width=min(400, image.shape[1]))
    (squares, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
    for (x, y, w, h) in squares:
        cv.rectangle(image, (x, y), (x + w, y + h),(100, 0, 255), 1)
        cv.imshow("Detecting people", image)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
