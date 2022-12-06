import cv2
import numpy as np

image_path = "images\letters.jpg"
image = cv2.imread(image_path, 0)
cv2.imshow("original image", image)
cv2.waitKey()

# Função que implementa Laplaciano "Padrão" - Saídas negativas incluídas
image_lapl = cv2.Laplacian(image, ddepth=cv2.CV_64F, ksize=1)
image_lapl_a = np.uint8(np.absolute(image_lapl))
cv2.imshow("laplacian Ab", image_lapl_a)
cv2.waitKey()

# Transformação para visualização, somando mínimo da função Laplaciano
image_lapl2 = image_lapl + image_lapl.min()
image_lapl2 = (image_lapl2 - image_lapl2.max()) / (image_lapl2.min() - image_lapl2.max())
image_lapl2 = (image_lapl2 * 255).astype(np.uint8)
cv2.imshow("laplacian Gray", image_lapl2)
cv2.waitKey()

# Função que implementa Laplaciano "Padrão" - Saída já para visualização, sem negativos
image_lapl = cv2.Laplacian(image, ddepth=-1, ksize=1)
cv2.imshow("laplacian", image_lapl)
cv2.waitKey()

# Para implementação da variação (com diagonais)
kernel = np.array([
  [ 0,  1,  0],
  [ 1, -4,  1],
  [ 0,  1,  0]
])

image_lapl1 = cv2.filter2D(image, -1, kernel)
cv2.imshow("laplacian with kernel", image_lapl1)
cv2.waitKey()

c = -1
inver = c*(image_lapl_a)
inver = (inver - inver.max()) / (inver.min() - inver.max())
inver = (inver * 255).astype(np.uint8)

iamge_agu = image + inver
cv2.imshow("Realce", iamge_agu)
cv2.waitKey()



