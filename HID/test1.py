from PIL import Image

img = Image.open("test1.png")
rotate = 90
img = img.rotate(angle=rotate,expand=True,resample=Image.BICUBIC)

img.show()