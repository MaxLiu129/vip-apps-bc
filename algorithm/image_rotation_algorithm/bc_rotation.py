import numpy as np
#from algo_lib import otsu_threshold, median_filter, threshold
#from test import neuralNetwork
import matplotlib.pyplot as plt
import math

import statistics as stat
import sys
#import PIL
#from PIL import Image

# todo:
# 1. gray scale the pic
# 2. find the edge
# 3. rotate the image to the right angle
# 4. figure out what does the bc says.

# this function will convert the pic into grayscale
def gray_scale(img_file, show_hist):
    img = (0.2126 * img_file[:,:,0] + 0.7152 * img_file[:,:,1] + 0.0722 * img_file[:,:,2])
    plt.figure()
    plt.imshow(img, cmap = "gray")
    # if we want to see the hist of the grayscaled 
    #if show_hist == 1:
    
    # write the pixels data into a txt (for test)
    textfile = open("img_pixel.txt", "w")
    for row in img:
        np.savetxt(textfile, row)
    textfile.close()
    plt.show()
    
    
# this function will detect the image edge.
def edge_detection(image, threshold):
    pass

# this function will turn all input img from a multi-color one to a black and white one.
def black_white_img(img_file):
    for 
    pass

# this function will tell if the px is a black pixel or no.
# We will not restrictly set the thershold to 0 as black but something
# close to 0
# Input: 
# 1. pixel: the pixel data which will contain 4 numbers like [0. 0. 0. 1.] for a black point?
# Output:
# 1. True if the px is black
#    False if the px is not black.
def black_px_detection(pixel):
    for i in range(2):
        if pixel[i] >= 0.25:
            return False
    return True


# this is the function that will meause the point which 
# from 'one of the pixel at the top of the image'
# to 'the top edge of actual business card'
# input args:
# 1. img_file: this is the raw img_file read using PIL
# 2. start_pixel: this is the position (index) of the starting pixel
#                 at the top of the img
# 3. thickness: this act like a thershold incase in the future
#               there might be some noise to effect the result.
#               So only when we detect some numbers of all black pixels
#               in a vertical line, we call it detects the top edge of the 
#               business card successfully.
# Output:
# 1. len: which will be the length of the pixels from the top of the img to 
#         the top of the business card.

# notes: we assume the input img_file has already been modified 
# and only contains PURE BLACK and WHITE pixels (10/16/2021) 
def length_to_top(img_file, start_pixel, thickness):
    len = 0 # set this up only for testing reason
    h = 100 # we know the height of the pic for testing, we will have a function later(10/16/2021)
    px_list = img_file[:, start_pixel, :]
    
    # detect process:
    # 1. iterrate from top to the bottom of the img
    # 2. detect a black pixel!
    #    2.1. start a loop of 5 times. make sure we have 5 black pts in a row
    #    2.2. yes, we have it. record the len and return
    #    2.3. No, it is a noise. go back to 1. loop
    count = 0
    len = 0
    for i in range(h):
        current_pixel = px_list[i]
        # this function will return true if the pixel is black
        if black_px_detection(current_pixel):
            #print("got one")  #only testing
            count += 1
        else:
            count = 0
        if count >= 5:
            len = i
            break
    if len == 0:
        print("len == 0, no black edge found!")
        return 0
    else:
        return len
    

# This function will find the angel of a img that 
def angel_finder(img_file):
    # 1. find the height and width of the img.
    # for this one, we assume we have embeded function for that.
    w = 200
    h = 100
    
    # 2. find two points to detect the edge point:
   
    px_1_ind = 66
    px_2_ind = 133
    while(px_1_ind < px_2_ind){
        # 2.1 start to search until it hit a black pixel for 5 times? (thershold is changable)
        thickness = 3 # thickness represent the times that black pixels 
        len1 = length_to_top(img_file, px_1_ind, thickness)   # this function needs to be done
        len2 = length_to_top(img_file, px_2_ind, thickness)
        print(len1, "what?",  len2)
        # 3. calculation
        angel = 0 # just a default value, delete when finished.
        angel = math.degrees(math.atan((abs(len1-len2) / (px_2_ind - px_1_ind))))
        if(len1 == 0 | len2 == 0):
            px_1_ind += 1
            px_2_ind -= 1
            print("no edge found when px")
    print(angel)
    }                     
    # 4. rotation
    return angel

# this is the function that will only rotation the img with
# specific angel. (clockwisely)
# Input: 
# 1. img_file: raw img file
# 2. angel: angel that the pic will be rotated ANTI-clockwisely
#          (in degree) so we need to make this to negative.
# Output:
# 1. img_rotated: rotated img file.
def rotate_img(img_file_name, angel):
    angel *= -1
    original = Image.open(img_file_name)
    rotated_img = original.rotate(angel)
    rotated_img.save('200x100_rotated.png')

    #return rotate_img


def angel_detection(img_file):
    pass

if __name__ == "__main__":
    img_file = plt.imread("test_img/real_test1.png")
    #img_file_pil = 
    #img_file[0][0][3] = 0
    
    # count = 0
    # for i in range(len(img_file[:,:,0])):
    #     if img_file[0,0,i] == 0:
    #         count+=1
    # print(count)
    
    
    
    # testing the function: black_pixel_detection
    pixel_test1 = img_file[:, 1, :][0]
    #print(black_px_detection(pixel_test1))

    # testing the function: rotation
    angel = (angel_finder(img_file))

    # testing rotation function:
    rotate_img("test_img/real_test1.png", angel)
    # testing the function: gray_scale
    #gray_scale(img_file, 1)
    
    
    
