c = -1
inver = c*(image_lapl)
inver = (inver - inver.max()) / (inver.min() - inver.max())
inver = (inver * 255).astype(np.uint8)

iamge_agu = image + inver
cv2.imshow("Aguçada", iamge_agu)
cv2.waitKey()