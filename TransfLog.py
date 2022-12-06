import cv2
import numpy as np

image_Path = "images\person.jpg"
image = cv2.imread(image_Path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

#c = 255 / np.log(1 + np.max(image))
c = 10
for i in range(163):
    for j in range(160):
        if image[i, j] == 255:
            image[i, j]= np.log(image[i, j])
        else:
            image[i, j] = np.log(1 + image[i, j])


imageLog = c * image
imageLog = np.array(imageLog, dtype=np.uint8)
cv2.imshow("Imagem Transformada", imageLog)
cv2.waitKey(0)