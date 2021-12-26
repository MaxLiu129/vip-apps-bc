import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.draw import disk
from skimage.morphology import erosion, dilation
# use opencv here, do not use this later
import cv2


def multi_erosion(image, kernel, iterations):
    for i in range(iterations):
        image = myerosion(image, kernel)   # switch to sidd's erosion function instead of using something from skimage.
    return image


if __name__ == "__main__":
    # Read the original image
    img = cv2.imread('test_img/002.jpg')               # change here when input changes

    # Display original image
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    
    # using opencv to pre-processing the pic first so we can do the erosion work
    edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=100) # Canny Edge Detection

    # Display Canny Edge Detection Image
    # plt.imshow(edges, cmap="gray")
    # plt.title("Canny Edge Detection Image")
    # plt.show()
    
    # this is a horizontal line
    h_line = np.array([[0,0,0,0,0],
                       [1,1,1,1,1],
                       [0,0,0,0,0]])
    short_line = np.array([[0,0],[1,1],[0,0]])
    
    longlong_line = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    
    long_line = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    v_line = np.array([[0,1,0],
                        [0,1,0],
                        [0,1,0]])
    
    h_short_line = np.array([[0,0,0],
                            [1,1,1],
                            [0,0,0]])
        
    small_block = np.array([[1,1,1], [1,1,1], [1,1,1]])
    # test if the h_line can be rendered properly.
    #imshow(h_line, cmap = 'gray');
    # start to test the erosion function from skimage library.
    eroded_img = erosion(edges, short_line)
    eroded_img2 = erosion(edges, v_line)
    
    # plt.imshow(h_short_line, cmap="gray")
    # plt.title("eroded_img")
    # plt.show()
    
    # plt.imshow(eroded_img, cmap="gray")
    # plt.title("eroded_img")
    # plt.show()
    
    # plt.imshow(eroded_img2, cmap="gray")
    # plt.title("eroded_img2")
    # plt.show()
    
    # plt.imshow(dilation_img1, cmap="gray")
    # plt.title("dilation_block")
    # plt.show()
    
    # first dilation the edges with blocks so the edges will be thinker and will be harder to be eroded.
    dilation_img1 = dilation(edges, small_block)
    # then we erode two times. see what is going to happen
    eroded_img3 = erosion(dilation_img1, long_line)
    
    dilation_img2 = dilation(eroded_img3, small_block)
    
    eroded_img4 = erosion(dilation_img2, long_line)
    
    # plt.imshow(dilation_img2, cmap="gray")
    # plt.title("eroded_longline")
    # plt.show()
    
    # plt.imshow(eroded_img4, cmap="gray")
    # plt.title("eroded_longline_twoErosion")
    # plt.show()
    
    im = Image.fromarray(eroded_img4)

    im.save("test_out/002_out.png")               # change here when input changes
    #eroded_img4.savefig('test_out/002_out.png', bbox_inches='tight', pad_inches=0)