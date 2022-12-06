import cv2
import numpy as np

image_path = "images\Roof.JPG"
image = cv2.imread(image_path)
cv2.imshow("original image", image)
cv2.waitKey()

kernel1 = np.array([
  [-2, -1,  0],
  [-1,  0,  1],
  [ 0,  1,  2]])

image_diag1 = cv2.filter2D(image, -1, kernel1)
cv2.imshow("Diagonal Prewitt", image_diag1)
cv2.waitKey()

kernel1 = np.array([
  [ 0,  1,  2],
  [-1,  0,  1],
  [-2, -1,  0]])

image_diag1 = cv2.filter2D(image, -1, kernel1)
cv2.imshow("Diagonal Prewitt 1", image_diag1)
cv2.waitKey()
