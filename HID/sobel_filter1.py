import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("bc_test1.jpg")

#vertical filter:
vertical_filter = [[-1,-2,-1],[0,0,0],[1,2,1]]
#horizontal filter:
hori_filter = [[-1,0,1],[-2,0,2],[-1,0,1]]

# get the width and height of the raw image
n,m,d = img.shape
#print(img)
# 
vertical_edges_img = np.zeros_like(img)

for row in range(3, n-2):
    for col in range(3, m-2):
        local_pixels = img[row-1:row+2, col-1:col+2, 0]
        transformed_pixels = vertical_filter*local_pixels
        vertical_score = (transformed_pixels.sum() + 4) / 8
        vertical_edges_img[row, col] = [vertical_score]*3
plt.imshow(vertical_edges_img, cmap='gray')


# convert numpy to img
#img = Image.fromarray(vertical_edge_img, 'RGB')

# saving an image in PIL#
#img.save('test1_result.png')
#img.show()
