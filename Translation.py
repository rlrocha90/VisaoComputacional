import cv2
import numpy as np

image_Path = "images\person.jpg"
image = cv2.imread(image_Path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

# Matriz de translação
translationMatrix = np.float32([[1, 0, 50], # Mover 50 pixels ao longo do eixo X (para a direita)
                                [0, 1, 20]]) # Mover 20 pixels ao longo do eixo Y (para baixo)

# Mover a imagem
movedImage = cv2.warpAffine(image, translationMatrix, (image.shape[1], image.shape[0]))

cv2.imshow("Imagem transladada", movedImage)
cv2.waitKey(0)