import cv2
import numpy as np

image_path = "images\pcb.jpeg"
image = cv2.imread(image_path)

# Máscara média
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]) / 9

# Utilizando a máscara definida no "kernel"
resulting_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_ISOLATED)

cv2.imshow("original image", image)
cv2.imshow("filter2d image", resulting_image)
cv2.waitKey()

# utilizando a função blud
resulting_image2 = cv2.blur(image, ksize=(3, 3))
cv2.imshow("filter2d image - mean", resulting_image2)
cv2.waitKey()
