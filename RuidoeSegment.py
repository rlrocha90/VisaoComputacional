import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\\Noise.png"
image1 = cv2.imread(image_path, 0)
cv2.imshow("Imagem Original", image1)
cv2.waitKey(0)

hist1 = cv2.calcHist([image1], [0], None, [256], [0, 255])
plt.plot(hist1)
plt.show()

(T, imageBin) = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(T)
cv2.imshow("Binarizado original", imageBin)
cv2.waitKey(0)

resulting_image1 = cv2.GaussianBlur(image1, ksize=(11, 11), sigmaX=9, sigmaY=9)
cv2.imshow("filter2d image - median", resulting_image1)
cv2.waitKey()

hist2 = cv2.calcHist([resulting_image1], [0], None, [256], [0, 255])
plt.plot(hist2)
plt.show()

(T, imageBin) = cv2.threshold(resulting_image1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(T)
cv2.imshow("Binarizado Suzavizado", imageBin)
cv2.waitKey(0)



