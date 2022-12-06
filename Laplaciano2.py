import cv2
import numpy as np

image_path = "images\Dot1.JPG"
image = cv2.imread(image_path, 0)
cv2.imshow("original image", image)
cv2.waitKey()

# Para implementação da variação (com diagonais)
kernel = np.array([
  [ 1,  1,  1],
  [ 1, -8,  1],
  [ 1,  1,  1]
])

image_lapl1 = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=kernel)
image_lapl1 = np.uint8(np.absolute(image_lapl1))
cv2.imshow("laplacian with kernel", image_lapl1)
cv2.waitKey()

x, y = image_lapl1.shape

for i in range(x):
    for j in range(y):
        if image_lapl1[i][j] > image_lapl1.max() - 1:
            image_lapl1[i][j] = 255
        else:
            image_lapl1[i][j] = 1

cv2.imshow("Limiarizado", image_lapl1)
cv2.waitKey()


