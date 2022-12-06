import cv2
import math
import numpy as np

image_path = "images\Bone.tif"
image1 = cv2.imread(image_path)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

i = 0
cnt = contours[i]

leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
print("Coordenada mais a esquerda: ", leftmost)
print("Coordenada mais a direita: ", rightmost)
print("Coordenada mais acima: ", topmost)
print("Coordenada mais abaixo: ", bottommost)

diametro = math.sqrt(((bottommost[0] - topmost[0]) ** 2) + ((bottommost[1] - topmost[1]) ** 2))
perimetro = cv2.arcLength(cnt, True)
print("perimetro da fronteira: ", perimetro)
print("Diametro - ponto mais alto - ponto mais baixo: ", diametro)

x, y, w, h = cv2.boundingRect(cnt)
print("Ponto mais a esquerda, em x: ", x)
print("Ponto mais alto, em y: ", y)
print("Largura: ", w)
print("Altura: ", h)

cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 255, 0), 2)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image1, [box], 0, (0, 0, 255), 2)

cv2.imshow("rec", image1)
cv2.waitKey()

(x1, y1), (MA, ma), angle = cv2.fitEllipse(cnt)
print("x do centro: ", x1, " e y do centro: ", y1)
print("Largura ou comprimento do eixo menor: ", MA)
print("Altura ou comprimento do eixo maior: ", ma)
print("Angulo de rotacao (clockwise): ", angle)

excentricidade = ma / MA
print("Excentididadae: ", excentricidade)
