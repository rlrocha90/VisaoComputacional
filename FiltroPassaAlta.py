import cv2
import numpy as np

image_path = "images\person.JPG"
image = cv2.imread(image_path)
cv2.imshow("original image", image)
cv2.waitKey(0)

#Definir Kernel Passa Alta
kernel = np.array([
    [-1,  -1, -1],
    [-1,   8, -1],
    [-1,  -1, -1],
])

image_PA = cv2.filter2D(image, -1, kernel)
cv2.imshow("Imagem filtrada", image_PA)
cv2.waitKey(0)

image_PA_neg = 255 - image_PA
cv2.imshow("Imagem filtrada Neg", image_PA_neg)
cv2.waitKey(0)
