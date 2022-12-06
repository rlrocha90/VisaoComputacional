import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images\Levedura.tif"
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

kernel = np.array([
  [ 0,  1,  0],
  [ 1, -4, 1],
  [ 0,  1,  0]
])

image_lapl1 = cv2.filter2D(image1, ddepth=cv2.CV_64F, kernel=kernel)
image_lapl1 = np.uint8(np.absolute(image_lapl1))
cv2.imshow("laplacian with kernel", image_lapl1)
cv2.waitKey()

image_lapl1 = ((image_lapl1-image_lapl1.min()) * (255/(image_lapl1.max()-image_lapl1.min())))
image_lapl1 = np.uint8(np.absolute(image_lapl1))
#cv2.imshow("Reesc", image_lapl1)
#cv2.waitKey()


(T, image_lapl1) = cv2.threshold(image_lapl1, 80, 255, cv2.THRESH_BINARY)

image_lapl1 = np.uint8(np.absolute(image_lapl1))
cv2.imshow("Limiarizado", image_lapl1)
cv2.waitKey()

imm = image_lapl1 * 2 * image1

cv2.imshow("Produto", imm)
cv2.waitKey()

hist2 = cv2.calcHist([imm], [0], None, [256], [1, 255])
plt.plot(hist2)
plt.show()

(T, imageBin1) = cv2.threshold(imm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(T)

(T, imageBin1) = cv2.threshold(image1, T, 255, cv2.THRESH_BINARY)
print(T)
cv2.imshow("Binarizado Bordas", imageBin1)
cv2.waitKey(0)



