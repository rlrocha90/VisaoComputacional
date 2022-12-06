# Program: Loading, Exploring  and Showing an Image

import cv2

image_path = "images\LogoInatel.png"
image = cv2.imread(image_path)
print("Dimensão da imagem: ", image.ndim)
print("Altura da imagem: ", format(image.shape[0]))
print("Largura da imagem: ", format(image.shape[1]))
print("Canais da imagem: ", format(image.shape[2]))
print("Tamanho do vetor imagem", image.size)
cv2.imshow("Minha imagem", image)
cv2.waitKey(0)

