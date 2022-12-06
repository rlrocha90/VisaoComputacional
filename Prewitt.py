import cv2
import numpy as np

image_path = "images\pcb1.png"
image = cv2.imread(image_path)
cv2.imshow("original image", image)
cv2.waitKey()

# Prewitt com derivada ao longo de x
kernelx = np.array([
  [-1, -1, -1],
  [ 0,  0,  0],
  [ 1,  1,  1]
])
image_prewittx = cv2.filter2D(image, -1, kernelx)
cv2.imshow("Sobel X1", image_prewittx)
cv2.waitKey()


# Prewitt com derivada ao longo de y
kernely = np.array([
  [-1,  0,  1],
  [-1,  0,  1],
  [-1,  0,  1]
])
image_prewitty = cv2.filter2D(image, -1, kernely)
cv2.imshow("Sobel Y", image_prewitty)
cv2.waitKey()