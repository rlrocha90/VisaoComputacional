import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\Contorno.tif"
image1 = cv2.imread(image_path)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

resulting_image = cv2.GaussianBlur(image, ksize=(9, 9), sigmaX=9, sigmaY=9)
cv2.imshow("filter2d image - Gaussian", resulting_image)
cv2.waitKey()

(T, imageBin) = cv2.threshold(resulting_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Binarizado Suzavizado", imageBin)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(imageBin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
print(contours)
image_copy = image1.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=0, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

cv2.imshow("Contorno None", image_copy)
cv2.waitKey()

contours1, hierarchy1 = cv2.findContours(imageBin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
print(contours)
image_copy = image1.copy()
cv2.drawContours(image=image_copy, contours=contours1, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

cv2.imshow("Contorno Simple", image_copy)
cv2.waitKey()

