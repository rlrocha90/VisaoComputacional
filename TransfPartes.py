import cv2
import numpy as np

image_Path = "images\polen.JPG"
image = cv2.imread(image_Path, 0)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

def pixelVal(img, r1, s1, r2, s2):
    if (0 <= img and img <= r1):
        return (s1 / r1) * img
    elif (r1 < img and img <= r2):
        return ((s2 - s1) / (r2 - r1)) * (img - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (img - r2) + s2

minimo = image.min()
print("Intensidade mínima: ",  minimo)
maximo = image.max()
print("Intensidade máxima: ", maximo)

# Parâmetros das partes
r1 = minimo #mínimo da faixa da imagem
s1 = 0
r2 = maximo # Máximo da faixa da imagem
s2 = 255

pixelVal_vec = np.vectorize(pixelVal)
imagePart = pixelVal_vec(image, r1, s1, r2, s2)
imagePart = np.array(imagePart, dtype=np.uint8)
cv2.imshow("Imagem Transformada", imagePart)
cv2.waitKey(0)
