import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey_frame, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            print("Componente na posição: ", x, " e ", y, " fora das dimensões...")
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.putText(frame, str(area), (x+70, y), 0, 1, (255, 0, 0))



    cv2.imshow("Frame", frame)
    cv2.imshow("Grey Frame", grey_frame)
    cv2.imshow("Limiarizado", binary)
    key = cv2.waitKey(5)
    if key ==27: #ESC
        break


cap.release()
cv2.destroyAllWindows()