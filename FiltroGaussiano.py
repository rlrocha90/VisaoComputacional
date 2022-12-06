import cv2
import numpy as np

image_path = "images\pcb.jpeg"
image = cv2.imread(image_path)
cv2.imshow("original image", image)
cv2.waitKey()

resulting_image1 = cv2.GaussianBlur(image, ksize=(7, 7), sigmaX=7, sigmaY=7)
cv2.imshow("filter2d image - median", resulting_image1)
cv2.waitKey()