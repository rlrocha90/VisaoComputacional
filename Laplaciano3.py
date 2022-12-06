import cv2
import numpy as np

image_path = "images\pcb1.png"
image = cv2.imread(image_path, 0)
cv2.imshow("original image", image)
cv2.waitKey()

# Função que implementa Laplaciano "Padrão" - Saídas negativas incluídas
image_lapl = cv2.Laplacian(image, ddepth=cv2.CV_64F, ksize=1)
image_lapl1 = np.uint8(np.absolute(image_lapl))
image_lapl3 = cv2.Laplacian(image, ddepth=-1, ksize=1)

kernel = np.array([
  [ 1,  1,  1],
  [ 1, -8,  1],
  [ 1,  1,  1]
])
#image_lapl = cv2.filter2D(image, ddepth=cv2.CV_64F, kernel=kernel)
# Transformação para visualização, somando mínimo da função Laplaciano
image_lapl2 = image_lapl + image_lapl.min()
image_lapl2 = (image_lapl2 - image_lapl2.max()) / (image_lapl2.min() - image_lapl2.max())
image_lapl2 = (image_lapl2 * 255).astype(np.uint8)
cv2.imshow("laplacian Gray", image_lapl2)
cv2.waitKey()

# Desconsiderando as saídas negativas - Detecção de linhas
x, y = image_lapl.shape

for i in range(x):
    for j in range(y):
        if image_lapl[i][j] < 0:
            image_lapl[i][j] = 0


cv2.imshow("laplacian sem valores negativos", image_lapl.astype(np.uint8))
cv2.waitKey()
cv2.imshow("absoluto", image_lapl1)
cv2.waitKey()
cv2.imshow("laplacian", image_lapl3)
cv2.waitKey()