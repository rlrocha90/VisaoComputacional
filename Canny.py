import cv2
import numpy as np

image_path = "images\letters.jpg"
image = cv2.imread(image_path, 0)
cv2.imshow("original image", image)
cv2.waitKey()

image_canny = cv2.Canny(image, 100, 200)
cv2.imshow("Canny", image_canny)
cv2.waitKey()
