import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\Bubble.PNG"

image1 = cv2.imread(image_path, 0)
cv2.imshow("Imagem Original", image1)
cv2.waitKey(0)

plt.hist(image1, 4, rwidth=0.9)
plt.show()

(T, imageBin) = cv2.threshold(image1, 170, 255, cv2.THRESH_BINARY)
print(T)
cv2.imshow("Binarizado", imageBin)
cv2.waitKey(0)
