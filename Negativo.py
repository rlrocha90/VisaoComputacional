import cv2
import numpy as np

image_Path = "images\person.jpg"
image = cv2.imread(image_Path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

negative = image.max() - image
cv2.imshow("Imagem Negativo", negative)
cv2.waitKey(0)


