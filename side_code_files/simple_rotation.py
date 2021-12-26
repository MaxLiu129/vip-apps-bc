from PIL import Image

img = Image.open("bc_test1.jpg")

rotated = img.rotate(10, fillcolor='gray')

rotated.show()