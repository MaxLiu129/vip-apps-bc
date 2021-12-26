from PIL import Image
import numpy as np

img = Image.open("test1.png")

#vertical filter:
ver_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
#horizontal filter:
hori_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]

# get the width and height of the raw image
w,h = img.size

# 
vertical_edge_img = np.zeros_like(img)
img_pixels = img.load()
print(img)
# for row in range(3, w-2):
#     for col in range(3, h-2):
#         local_pixels = img_pixels[row-1:row+2, col-1:col+2]
#         transformed_pixels = vertical_filter*local_pixels
#         vertical_score = (transformed_pixels.sum() + 4)/8
#         vertical_edges_img[row, col] = [vertical_score] * 3



# convert numpy to img
img = Image.fromarray(vertical_edge_img, 'RGB')

# saving an image in PIL
#img.save('test1_result.png')
img.show()
