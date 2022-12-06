import cv2
import numpy as np

image_Path = "images\person.jpg"
image = cv2.imread(image_Path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

# Recorte a imagem (necessário um "norte" de onde e como recortar)
croppedImage = image[40:120, 40:120]
cv2.imshow("Imagem cortada", croppedImage)
cv2.waitKey(0)