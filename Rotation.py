import cv2
import numpy as np

image_Path = "images\person.jpg"
image = cv2.imread(image_Path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

# formato da imagem. Altura, largura e canais
(h, w) = image.shape[:2] # (h, w, c) = image.shape[:]

# Matriz de rotação
center = (h//2, w//2)
angle = -45
scale = 1.0

rotationMatrix = cv2.getRotationMatrix2D(center, angle, scale)

# Rotacionar imagem
rotatedImage = cv2.warpAffine(image, rotationMatrix, (image.shape[1], image.shape[0]))

cv2.imshow("Imagem Rotacionada", rotatedImage)
cv2.waitKey(0)