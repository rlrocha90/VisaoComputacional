import cv2
import numpy as np

image_Path = "images\Forest.JPG"
image = cv2.imread(image_Path, 0)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

c = 1
gamma = 0.7

imagePot = np.array(255*(image / 255) ** gamma, dtype = 'uint8')
#imagePot8 = np.array(imagePot, dtype=np.uint8)
cv2.imshow("Imagem Transformada", imagePot)
cv2.waitKey(0)