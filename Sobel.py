import cv2
import numpy as np

image_path = "images\pcb1.png"
image = cv2.imread(image_path)
cv2.imshow("original image", image)
cv2.waitKey()

# Sobel com derivada ao longo de x
image_sobelx = cv2.Sobel(image, ddepth=-1, dx=1, dy=0, ksize=3)
cv2.imshow("Sobel X", image_sobelx)
cv2.waitKey()

# Sobel com derivada ao longo de y
image_sobely = cv2.Sobel(image, ddepth=-1, dx=0, dy=1, ksize=3)
cv2.imshow("Sobel Y", image_sobely)
cv2.waitKey()