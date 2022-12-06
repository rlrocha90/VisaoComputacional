import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\pcb.jpeg"

image1 = cv2.imread(image_path, 0)
cv2.imshow("Imagem Original", image1)
cv2.waitKey(0)

plt.hist(image1, 4, rwidth=0.9)
plt.show()

# Adaptativo com média dos pixels ao redor, baseado no valor de vizinhança inserido na função
image_bin = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
cv2.imshow("Binarizada mean", image_bin)
cv2.waitKey(0)

# Adaptativo com média ponderada de todos os pixels ao redor, baseado no valor de vizinhança inserido na função
image_bin = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)
cv2.imshow("Binarizada gaussian", image_bin)
cv2.waitKey(0)