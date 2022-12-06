import cv2
import numpy as np

image_Path = "images\person.jpg"
image = cv2.imread(image_Path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

# Espelhamento horizontal
flippedHorizontally = cv2.flip(image, 1)
cv2.imshow("Espelhamento horizontal", flippedHorizontally)
cv2.waitKey(-1)

# Espelhamento vertical
flippedVertically = cv2.flip(image, 0)
cv2.imshow("Espelhamento vertical", flippedVertically)
cv2.waitKey(-1)

# Espelhamento horizontal e depois vertical
flippedHV = cv2.flip(image, -1)
cv2.imshow("Espalhamento H e V", flippedHV)
cv2.waitKey(-1)
