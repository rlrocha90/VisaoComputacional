import cv2
import numpy as np

image_path = "images\person.JPG"
image = cv2.imread(image_path)
cv2.imshow("original image", image)
cv2.waitKey(0)

#Definir Kernel Passa Baixa
kernel = (1/8) * np.array([
    [0,  1, 0],
    [1,  4, 1],
    [0,  1, 0],
])

image_PB = cv2.filter2D(image, -1, kernel)
cv2.imshow("Imagem filtrada", image_PB)
cv2.waitKey(0)
