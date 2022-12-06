# Program: OpenCV code to access and manupulate pixels

import cv2

image_path = "images\LogoInatel.png"
image = cv2.imread(image_path)

# Acessar pixel na localização (0,0)
(b, g, r) = image[0, 0]
print("Blue, Green and Red na posição (0,0): ", format((b, g, r)))

# manipular pixel e mostrar imagem modificada
# Importante: Ao acessar o Arranjo nas 3 dimensões, temos BGR e não RGB.
# Importante 2: Para manipular, (b, g, r) e não (r, g, b) valores

image[30:70, 100:150] = (0, 0, 255)
cv2.imshow("Imagem modificada", image)
cv2.waitKey(0)