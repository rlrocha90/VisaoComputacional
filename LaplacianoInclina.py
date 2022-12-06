import cv2
import numpy as np

image_path = "images\Pcb2.tif"
image = cv2.imread(image_path, 0)
cv2.imshow("original image", image)
cv2.waitKey()

# Para implementação da variação (com diagonais)
kernel = np.array([
  [ 2, -1, -1],
  [-1,  2, -1],
  [-1, -1,  2]
])

image_lapl1 = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=kernel)
x, y = image_lapl1.shape

for i in range(x):
    for j in range(y):
        if image_lapl1[i][j] < 0:
            image_lapl1[i][j] = 0


cv2.imshow("laplacian +45", image_lapl1)
cv2.waitKey()

kernel = np.array([
  [-1, -1,  2],
  [-1,  2, -1],
  [ 2, -1, -1]
])

image_lapl2 = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=kernel)
x, y = image_lapl2.shape

for i in range(x):
    for j in range(y):
        if image_lapl2[i][j] < 0:
            image_lapl2[i][j] = 0
cv2.imshow("laplacian -45", image_lapl2)
cv2.waitKey()

x, y = image_lapl1.shape

for i in range(x):
    for j in range(y):
        if image_lapl1[i][j] > image_lapl1.max() - 1:
            image_lapl1[i][j] = 255
        else:
            image_lapl1[i][j] = 0

cv2.imshow("Limiarizado", image_lapl1)
cv2.waitKey()


