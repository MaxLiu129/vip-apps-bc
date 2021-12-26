import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import statistics as stat
import sys

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
        if pixel == 1:
            return False
    return True


# This function will find the angel of a img that 
def angel_finder(img_file):
    # 1 we need to start a loop, starting from the first column of the img to the 3/4 of the img?
    # 1.1 calculate the width and the height of the img
    h, w = img_file.shape
    
    # 1.2 start the big loop & initialize some VARs.
    good_h = int(h / 2)

    angel = 0
    px_cal_list = []
    px_list = []
    for i in range(0, w):
        current_j = 0
        new_j = 0
        px_list = img_file[:, i]
        counter = 0
        for j in range(0, h):
            # if it is not black px record it until we hit the next one.
            if not black_px_detection(px_list[j]):
                new_j = j
                # now we need to calculate the distance between new_j and current_j
                if new_j - current_j >= good_h:
                    px_cal_list.append([i,current_j])
                current_j = new_j
                counter = 0      
    #print(px_cal_list)
        
    # 2. now we would like to fetch the data that we want to calculate the angel
    final_px_list = []
    for pair in px_cal_list:
        if pair[1] != 0:
            final_px_list.append(pair)
    #print(final_px_list)
    # 3. we used the 3/8 and the 5/8 position of the final list pairs to calculate the angel here (idk if it is appliable for other imgs)
    len_1 = int(len(final_px_list) * 3 / 8)
    len_2 = int(len(final_px_list) * 5 / 8)

    angel = math.degrees(math.atan((final_px_list[len_1][1] - final_px_list[len_2][1]) 
                                    / abs(final_px_list[len_1][0] - final_px_list[len_2][0])))
    if (final_px_list[len_1][1] - final_px_list[len_2][1]) > 0:
        angel *= -1 
    print(angel)
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
    #angel *= -1
    original = Image.open(img_file_name)
    rotated_img = original.rotate(angel)
    #rotated_img.save('test_out/002_rotated.png')               # change here when input changes
    return rotated_img

def rotate(filepath, original_path):
    # import the img that we are going to calculate
    img_file = plt.imread(filepath)                   # change here when input changes

    # testing the function: angel_finder
    angel = (angel_finder(img_file))

    # testing rotation function:
    return rotate_img(original_path, angel)               # change here when input changes
