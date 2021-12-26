import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def ave(img_file):
    # for this function, mainly I would like to calculate the ave height of the top edge of the BC(business card)
    # Hopefully we will have a good result.
    # 1. irritate the width of the image, and detect the first white point that ever being touched
    w,h,d = img_file.shape
    white_list = []
    print(h)
    for i in range(h):
        for j in range(w):
            for k in range(3):
                if img_file[j][i][k] >= 0.7:
                    white_list.append([i, j])
                    print([i,j])
                    break
            break
    pass




img_file = plt.imread("002_purecolored.png")
ave(img_file)

