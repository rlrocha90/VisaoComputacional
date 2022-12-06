import cv2

image_path = "images\Leaf.tif"
image1 = cv2.imread(image_path)
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

esqueleto = cv2.ximgproc.thinning(image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
cv2.imshow("Imagem Esqueletizada", esqueleto)
cv2.waitKey(0)
