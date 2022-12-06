import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\Leaf.tif"
image1 = cv2.imread(image_path)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
i = 0
hull = cv2.convexHull(contours[i], False)

drawing = np.zeros((image1.shape[0], image1.shape[1], 3), np.uint8)
cv2.drawContours(drawing, contours, i, (0, 255, 0), -1, 8)
cv2.drawContours(drawing, hull, -1, (255, 0, 0), 3, 8)


cv2.imshow("Contorno Poligono", drawing)
cv2.waitKey()
