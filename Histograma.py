import cv2
import numpy as np
from matplotlib import pyplot as plt

image1 = cv2.imread("images\polenEscuro.JPG", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Escura", image1)
cv2.waitKey(0)
image2 = cv2.imread("images\polenClaro.JPG", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Clara", image2)
cv2.waitKey(0)
image3 = cv2.imread("images\polenBaixoContraste.JPG",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Baixo Contraste", image3)
cv2.waitKey(0)
image4 = cv2.imread("images\polenAltoContraste.JPG", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Alto Constraste", image4)
cv2.waitKey(0)

# Calcular o histograma
hist1 = cv2.calcHist([image1], [0], None, [256], [0, 255])
hist2 = cv2.calcHist([image2], [0], None, [256], [0, 255])
hist3 = cv2.calcHist([image3], [0], None, [256], [0, 255])
hist4 = cv2.calcHist([image4], [0], None, [256], [0, 255])

fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2
fig.add_subplot(rows, columns, 1)
plt.plot(hist1)
fig.add_subplot(rows, columns, 2)
plt.plot(hist2)
fig.add_subplot(rows, columns, 3)
plt.plot(hist3)
fig.add_subplot(rows, columns, 4)
plt.plot(hist4)
plt.show()

# Equalizar o histograma
equalizedImage = cv2.equalizeHist(image1) #Trocar a imagem para ver as equalizações
cv2.imshow("Imagem Equalizada", equalizedImage)
cv2.waitKey(0)
# Calcular o histograma da imagem equalizada
histEqualized = cv2.calcHist([equalizedImage], [0], None, [256], [0, 255])

fig = plt.figure(figsize=(10, 7))
plt.plot(histEqualized)
plt.xlabel("Bins")
plt.ylabel("Número de Pixels")
plt.show()