import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\Bone.tif"
image1 = cv2.imread(image_path)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

print(contours)
image_copy = image1.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

cv2.imshow("Contorno None", image_copy)
cv2.waitKey()

i = 0
cnt = contours[i]
peri = cv2.arcLength(contours[i], True)
approx = cv2.approxPolyDP(contours[i], 0.03 * peri, closed=True)

print("Quantidade de pontos: ", len(approx))
image_copy1 = image1.copy()
cv2.drawContours(image=image_copy1, contours=approx, contourIdx=-1, color=(0, 0, 255), thickness=3)

cv2.imshow("Contorno Poligono", image_copy1)
cv2.waitKey()

