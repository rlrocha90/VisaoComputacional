import cv2
import numpy as np

image_path = "images\pcb.jpeg"
image = cv2.imread(image_path)
cv2.imshow("original image", image)
cv2.waitKey()

resulting_image1 = cv2.medianBlur(image, ksize=3)
cv2.imshow("filter2d image - median", resulting_image1)
cv2.waitKey()