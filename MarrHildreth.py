import cv2
import numpy as np

image_path = "images\letters.jpg"
image = cv2.imread(image_path, 0)
cv2.imshow("original image", image)
cv2.waitKey()

# Para implementação da variação (com diagonais)
kernel = np.array([
  [ 0,  0, -1,  0,  0],
  [ 0, -1, -2, -1,  0],
  [-1,- 2, 16, -2, -1],
  [ 0, -1, -2, -1,  0],
  [ 0,  0, -1,  0,  0],
])

image_lapl1 = cv2.filter2D(image, -1, kernel)
cv2.imshow("MarrHildreth", image_lapl1)
cv2.waitKey()

x, y = image_lapl1.shape

for i in range(x):
    for j in range(y):
        if image_lapl1[i][j] > 200:
            image_lapl1[i][j] = 255
        else:
            image_lapl1[i][j] = 1

cv2.imshow("Limiarizado", image_lapl1)
cv2.waitKey()