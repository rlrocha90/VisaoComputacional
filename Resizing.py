import cv2
import numpy as np

image_Path = "images\person.jpg"
image = cv2.imread(image_Path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

# formato da imagem. Altura, largura e canais
(h, w) = image.shape[:2] # (h, w, c) = image.shape[:]
# calcular "aspect ratio"
aspect = w / h
print("Altura: ", h)
print("Largura: ", w)
print("Aspect Ratio: ", aspect)

# Redimensionar a imagem
height = int(0.5 * h)
widht = int(height * aspect)

print("Nova Altura: ", height)
print("Nova Largura: ", widht)
print("Novo Aspect Ratio: ", widht/height)

dimension = (height, widht)
resizedImage = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)
cv2.imshow("Imagem redimensionada", resizedImage)
cv2.waitKey(0)

# Usando fatores x e y
resizedWithFactors = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LANCZOS4)
cv2.imshow("Imagem redimensionada com fatores", resizedWithFactors)
cv2.waitKey(0)

