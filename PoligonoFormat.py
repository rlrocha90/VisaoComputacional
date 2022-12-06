import cv2
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\Formats.png"
image1 = cv2.imread(image_path)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

(T, imageBin) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(T)
cv2.imshow("Binarizado original", imageBin)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(imageBin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
print(contours)
image_copy = image1.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

cv2.imshow("Contorno None", image_copy)
cv2.waitKey()

i = 1 #Alterar para ver o resultado
peri = cv2.arcLength(contours[i], True)
approx = cv2.approxPolyDP(contours[i], 0.01 * peri, closed=True)

print("Quantidade de contornos: ", approx[0][0][0])
print("Quantidade de pontos: ", len(approx))
x = []
y = []
for a in range(len(approx)):
    x.append(approx[a][0][0])
    y.append(approx[a][0][1])


x = mean(x)
y = mean(y)
image_copy1 = image1.copy()

cv2.drawContours(image=image_copy1, contours=approx, contourIdx=-1, color=(0, 0, 255), thickness=3)
if len(approx) == 3:
    cv2.putText(image_copy1, "Tri", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
elif len(approx) == 4:
    cv2.putText(image_copy1, "Ret", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
elif len(approx) == 5:
    cv2.putText(image_copy1, "Pent", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
elif len(approx) == 6:
    cv2.putText(image_copy1, "Hex", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)


cv2.imshow("Contorno Poligono", image_copy1)
cv2.waitKey()