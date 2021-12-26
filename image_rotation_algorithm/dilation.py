import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.draw import disk
from skimage.morphology import erosion, dilation

## Siddharth Singh 11/11/21
## This function pads extra space onto the image
## as this makes future processes much simpler
def pad(img, kernel):
    X = img.shape[0] + kernel.shape[0] - 1
    Y = img.shape[1] + kernel.shape[1] - 1
    padded_image = np.zeros((X, Y))
    for i in range (0, img.shape[0]):
        for j in range(0, img.shape[1]):
            padded_image[i + 1, j + 1] = img[i, j]
    return padded_image

# this is the function that I copied from Sidd work.
def myerosion(img, kernel):
    padded_image = pad(img, kernel)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            window = padded_image[i:i + kernel.shape[0], j: j + kernel.shape[1]]
            compare = (window == kernel)
            if np.all(compare == True):
                img[i, j] = 1
            else:
                img[i, j] = 0
    return img


# we set up a circle in the middle of a pic with 100 x 100 px
# and the radius of the circle is 25.
circ_image = np.zeros((100, 100))
circ_image[disk((50, 50), 25)] = 1   

# render the image of the circle we just made.    
imshow(circ_image, cmap = "gray")

# set up a cross fig, in order to do the erosion process.
cross = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]])
imshow(cross, cmap = 'gray');

# erosion process here
eroded_circle = myerosion(circ_image, cross)
imshow(eroded_circle, cmap = 'gray');
plt.show()
# print the differences between erosion and non-erosion pic.
linecolor = 'red'
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(circ_image, cmap = 'gray');
ax[0].set_title('Original', fontsize = 19)
ax[0].axvline(x = 25, color = linecolor)
ax[0].axvline(x = 75, color = linecolor)
ax[0].axhline(y = 25, color = linecolor)
ax[0].axhline(y = 75, color = linecolor)

ax[1].imshow(eroded_circle, cmap = 'gray');
ax[1].set_title('Eroded', fontsize = 19)
ax[1].axvline(x = 25, color = linecolor)
ax[1].axvline(x = 75, color = linecolor)
ax[1].axhline(y = 25, color = linecolor)
ax[1].axhline(y = 75, color = linecolor)
fig.tight_layout()



# 
def multi_erosion(image, kernel, iterations):
    for i in range(iterations):
        image = myerosion(image, kernel)   # switch to sidd's erosion function instead of using something from skimage.
    return image

def multi_dilation(image, kernel, iterations):
    for i in range(iterations):
        image = dilation(image, kernel)
    return image



ites = [2,6,10,14,18,22]
fig, ax = plt.subplots(2, 3, figsize=(17, 5))
for n, ax in enumerate(ax.flatten()):
    ax.set_title(f'Iterations : {ites[n]}', fontsize = 16)
    new_circle = multi_erosion(circ_image, cross, ites[n])
    ax.imshow(new_circle, cmap = 'gray');
    ax.axis('off')
fig.tight_layout()
plt.show()

# set up a horizontal line and a vertical line for further use later.
h_line = np.array([[0,0,0],
                  [1,1,1],
                  [0,0,0]])
v_line = np.array([[0,1,0],
                  [0,1,0],
                  [0,1,0]])
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(h_line, cmap='gray');
ax[1].imshow(v_line, cmap='gray');
fig.tight_layout()

# if __name__ == "__main__":
#     complex_image = imread('test_edge/002_edge.png')
#     imshow(complex_image);
#     plt.show()
#     step_1 = multi_erosion(complex_image, h_line, 3)
#     step_2 = multi_erosion(step_1, v_line,3)
#     step_3 = multi_dilation(step_2, h_line,3)
#     step_4 = multi_dilation(step_3, v_line,3)
#     steps = [step_1, step_2, step_3, step_4]
#     names = ['Step 1', 'Step 2', 'Step 3', 'Step 4']
#     fig, ax = plt.subplots(2, 2, figsize=(10, 10))
#     for n, ax in enumerate(ax.flatten()):
#         ax.set_title(f'{names[n]}', fontsize = 22)
#         ax.imshow(steps[n], cmap = 'gray');
#         ax.axis('off')
#     fig.tight_layout()
#     plt.show()

