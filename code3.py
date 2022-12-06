from PIL import Image, ImageShow

pil_im = Image.open('LogoInatel.png')
ImageShow.show(pil_im)
pil_im_g = Image.open('LogoInatel.png').convert('L')
ImageShow.show(pil_im_g)
r, g, b = pil_im.getpixel((1, 1))
print(r, g, b)