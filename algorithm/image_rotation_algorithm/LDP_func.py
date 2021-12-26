import numpy as np
from algo_lib import otsu_threshold, median_filter, threshold
from test import neuralNetwork
from PIL import Image
import math
import statistics as stat


# Eli Coltin 3/7/21
# turns array into matrix
def arrayToMatrix(img, originalImg):
    imgLen = originalImg.size[0]
    imgWid = originalImg.size[1]
    matrix = np.reshape(img, (imgWid, imgLen))
    return matrix


# Eli Coltin 2/20/21
# Turns 2d matrix of pixels into image by flattening the matrix into an array
# and then putting that flat array into an image which it returns
# argument 0 is an image pixel matrix
# returns image

def matrixToImg(matrix):
    # flatten array to put into image
    flatArray = np.ndarray.flatten(matrix)
    # when putting flat array into image: (len, width)
    img = Image.new("L", [len(matrix[0]), len(matrix)])
    img.putdata(flatArray)
    return img


# Eli Coltin 2/27/21
# Create a white pixel sum graph for license plate image
# Find a threshold of values in pixel sum graph where anything below is probably space
# This threshold is determined using Otsu
# Then, set everything below that threshold to 0, everything above to the threshold
# argument 0 is an image array
# argument 1 is either horizontal or vertical: 0 for horizontal (row sum), 1 for vertical (col sum)
# argument 2 is the threshold of number of pixels in a row to indicate a character is there
# returns a array of if a pixel is part of a character or not
def threshGraph(img_array, ax, thresh=0):
    graph = np.sum(img_array, axis=ax)/255

    if(thresh == 0):
        thresh = otsu_threshold(graph)  # Eli Coltin 4/4/21
    threshGraph = []
    for i in graph:
        if i > thresh:
            threshGraph.append(thresh)
        else:
            threshGraph.append(0)
    return threshGraph


# Eli Coltin 4/4/21
# Create a white pixel sum graph for license plate image
# Find a threshold of values in pixel sum graph where anything below is probably space
# This threshold is determined using dividing from maximum value
# Then, set everything below that threshold to 0, everything above to the threshold
# argument 0 is an image array
# argument 1 is either horizontal or vertical: 0 for horizontal (row sum), 1 for vertical (col sum)
# argument 2 is the divisor which will be applied to the max value to determine threshold
# returns a array of if a pixel is part of a character or not
def threshGraphDivisor(img_array, ax, divisor):
    graph = np.sum(img_array, axis=ax)/255

    thresh = round(max(graph)/divisor)

    threshGraph = []
    for i in graph:
        if i > thresh:
            threshGraph.append(thresh)
        else:
            threshGraph.append(0)
    return threshGraph


# Eli Coltin 2/27/21
# Use the graph of space thresholds to determine where the start and ends of characters are
# This is done by determining where the parts above and below the threshold meet
# Add these character edges to a list
# Argument 0 is the thresholded graph of where the edges are
# returns a list of where each edge is
def edgeListCreator(graph):
    # create list for edges
    edgeList = []

    # if starting at char, append the edge
    if(graph[0] != 0):  # Eli Coltin 3/6/21
        edgeList.append(0)
    for i in range(0, len(graph)-1):
        if (graph[i] == 0) and (graph[i+1] != 0):
            edgeList.append(i)
        elif (graph[i] != 0) and (graph[i+1] == 0):
            edgeList.append(i)
    # if ending at char, append the edge
    if(graph[len(graph)-1] != 0):  # Eli Coltin 3/6/21
        edgeList.append(len(graph)-1)
    return edgeList


# Eli Coltin 2/22/21
# Find vertical character boundaries
# For each edge, add a buffer of 1% of the image. If it is thought to be a left buffer,
# add 1% to left, if right, add 1% to right
# if adding the buffer will go over image, set point to max length
# if adding buffer will go below 0, set point to 0
# argument 0 is the image of all the license plate charactesr
# argument 1 is the edgeList of the thresholded vertical pixel sum graph
# argument 2 is the amount of buffer
# returns a list of buffered edges for the characters
def verticalBoundaryFinder(img_array, edgeList, bufferMult=.01):
    upperBound = len(img_array[0])

    lines = []
    buffer = round(upperBound*bufferMult)
    even = 0

    for point in edgeList:
        if even == 1:
            if (point + buffer) < upperBound:
                lines.append(point + buffer)
            else:
                lines.append(upperBound)
            even = 0
        else:
            if (point - buffer) > 0:
                lines.append(point - buffer)
            else:
                lines.append(0)
            even = 1
    return lines


# Eli Coltin 3/7/21
# From processed image matrix, take slices of pixels where characters thought to be
# each part of matrix will be start and end, then tranform this into any array
# Argument 0: buffered edges from verticalBoundaryFinder
# Argument 1: license plate image used in verticalBoundaryFinder
# Argument 2: grayscale image # Eli Coltin 3/28/21
# Output: list of character locations, grayscale img
def characterListCreator(bufferedEdges, img_array, grayscale):
    characterList = []
    characterListGrayscale = []
    # iterate through each edge
    for i in range(0, len(bufferedEdges)-1, 2):
        # make range of where each character is, append to list
        start = bufferedEdges[i]
        end = bufferedEdges[i+1]
        # skip if width less than 5% of characters
        if(end-start) < len(img_array[0])*.05:
            continue
        characterList.append(img_array[:, start:end])
        characterListGrayscale.append(grayscale[:, start:end])

    # Eli Coltin 3/21/21
    # list of each character's length
    lenList = []
    # iterate through each character, append its length
    for char in characterList:
        lenList.append(len(char[0]))

    # Eli Coltin 3/28/21
    # if no characters, return empty
    if(len(lenList) == 0):
        return([], [])

    # find median length
    medLen = stat.median(lenList)

    # flag for if to delete character
    flag = False
    # look for blocks of characters still in character images
    # iterate through each character
    for i in range(len(characterList)):
        # if character length less than 1.75* median character's length
        if(len(characterList[i][0]) < 1.75*medLen):
            continue
        # else
        else:
            # character will be deleted
            flag = True
            # predict number of characters in block
            # divide length by median length
            numChar = round(len(characterList[i][0])/medLen)
            # if one character, no split chars
            if(numChar <= 1):
                break
            # find predicted distance between character
            # if evenly spaced
            distPerChar = math.floor(len(characterList[i][0])/numChar)
            for j in range(0, numChar):
                # make range of where each character is, append to list
                start = j*distPerChar
                end = j*distPerChar + distPerChar
                characterList.insert(i+j+1, characterList[i][:, start:end])
                characterListGrayscale.insert(
                    i+j+1, characterListGrayscale[i][:, start:end])
            characterList.pop(i)
            characterListGrayscale.pop(i)

    # if no characters were deleted, check a different case
    # if the minimum number is less than 2.5 times the maximum number
    if(flag == False and (min(lenList)*2.5 < max(lenList))):
        # do the whole process again, but with different threshold graph
        # raise division from 20 to 7.5
        horizGraph = threshGraphDivisor(img_array, 0, 7.5)
        return characterListCreator(verticalBoundaryFinder(img_array, edgeListCreator(horizGraph)), img_array, grayscale)

    # if only one character detected, but character not cropped at all
    # probably some issue, don't return any characters
    if(len(lenList) == 1 and lenList[0] == len(img_array[0])):
        return(([], []), ([], []))

    return(characterList, characterListGrayscale)


# Eli Coltin 3/7/21
# From processed image matrix, slice the left and right edges from license plate
# Argument 0: buffered edges from verticalBoundaryFinder
# Argument 1: license plate image used in verticalBoundaryFinder
# Argument 2: grayscale plate # Eli Coltin 3/28/21
# Argument 2: multiplication amount to buffer top, bottom edges of output plate: default is 0
# Argument 3: return image or upper, lower values of biggest chunk, if true (default) return image, false return tuple
# of lower, upper edge
# Output: sliced plate as matrix, grayscale
def horizontalImageSlicer(bufferedEdges, img_array, grayscale, bufferMult=0.0, returnImg=True):
    # take the upper and lower edges of the plate, slice between
    maxLen = 0
    # save start, end of maxLen
    maxStart = 0  # Eli Coltin 3/27/21
    maxEnd = 0  # Eli Coltin 3/27/21
    # height of image
    imgHeight = len(img_array)
    # buffer amount
    bufferAmt = round(imgHeight * bufferMult)

    # iterate through each connected peak/valley
    for i in range(0, len(bufferedEdges)-1):  # Eli Coltin 4/4/21 - look at all continuous
        start = bufferedEdges[i]
        end = bufferedEdges[i+1]
        if(end-start > maxLen):
            maxLen = end-start
            # Eli Coltin 3/27/21
            # store values
            maxStart = start
            maxEnd = end

    # Eli Coltin 3/27/21
    # Add buffer height to license plate
    if(maxStart - bufferAmt >= 0):
        maxStart = maxStart - bufferAmt
    if(maxEnd + bufferAmt < imgHeight):
        maxEnd = maxEnd + bufferAmt

    # Eli Coltin 3/27/21
    # return correct type
    if(returnImg):
        return(img_array[maxStart:maxEnd, :], grayscale[maxStart:maxEnd, :])
    else:
        return((maxStart, maxEnd))


# Eli Coltin 3/7/21
# From processed image matrix, slice the left and right edges graph
# Argument 0: buffered edges from verticalBoundaryFinder
# Argument 1: license plate image used in verticalBoundaryFinder
# Argument 2: grayscale plate # Eli Coltin 3/28/21
# Output: sliced plate as matrix, grayscale plate sliced
# Eli Coltin 4/4/21 Now can decide if slice vertical or horiz
def onlyEdgeSlicer(bufferedEdges, img_array, grayscale, vertical):
    # take the inner edges of the border of the plate, take the slice between
    start = bufferedEdges[1]
    end = bufferedEdges[len(bufferedEdges)-2]

    # Eli Coltin 4/4/21
    # changed it so that the focus is on only removing the left, right edges of graph, can do horiz or vertical
    if(vertical):
        return(img_array[:, start:end], grayscale[:, start:end])
    else:
        return(img_array[start:end, :], grayscale[start:end, :])


# Eli Coltin 2/27/21
# Use list of edgeList to determine where largest object is in vertical slice
# If there are no gaps in graph, return whole character
# Go through each edge gap, find the largest one, that is where the character is
# Make sure it is > 50% of original picture height to make sure it is actual character A-Z,0-9
# If it is not that size, returns empty list
# Otherwize, will return the image cut to horizontal boundaries
# add a 5% buffer to top, bottom
# argument 0 is the list of edges for the certain character
# argument 1 is the image
# argument 2 is the grayscale image # 3/28/21
# returns the character with buffered horizontal edges
def horizontalBoundaryFinder(edgeList, img, grayscale):
    if(len(edgeList) == 0):
        return(img, grayscale)

    upperBound = len(img)
    buffer = round(upperBound*.05)

    difMax = 0
    difLowerBound = 0
    difUpperBound = 0
    for i in range(0, len(edgeList)-1):
        dif = edgeList[i+1] - edgeList[i]
        if dif >= difMax:
            difMax = dif
            difLowerBound = edgeList[i]
            difUpperBound = edgeList[i+1]

    if(difUpperBound-difLowerBound < len(img)*.50):
        return([], [])

    if(difLowerBound - buffer > 0):
        difLowerBound = difLowerBound - buffer
    else:
        difLowerBound = 0

    if(difUpperBound + buffer < upperBound):
        difUpperBound = difUpperBound + buffer
    else:
        difUpperBound = upperBound

    return(img[difLowerBound:difUpperBound, :], grayscale[difLowerBound:difUpperBound, :])


# Eli Coltin 2/27/21
# For each vertical snip of where a character should be, now horizontally snip the character
# These characters vertically and horizontally snipped will be put into a new list
# If a character is determined to not be a correct character, it will not be saved
# argument 0 is list of characters
# argument 1 is grayscale list of characters # Eli Coltin 3/28/21
# returns list of trimmed characters, grayscale list

# may not be necessary
def characterTrimmer(characterList, grayscale):
    trimCharList = []
    grayscaleList = []
    for i in range(len(characterList)):
        graph = threshGraphDivisor(characterList[i], 1, 10)
        edgeList = edgeListCreator(graph)
        char = horizontalBoundaryFinder(
            edgeList, characterList[i], grayscale[i])
        if(len(char[0]) != 0):
            trimCharList.append(char[0])
            grayscaleList.append(char[1])
    return(trimCharList, grayscaleList)


# Eli Coltin 2/22/21
# Print where each character is supposed to be
# Create the amount of subplots for the thought amount of characters
# Then, output each character into a subplot
# argument 0 is list of characters
# argument 1 is if to save characters 3/13/21
# argument 2 is return type - if false, return the amount of images, if true, return the images # Eli Coltin 4/25/21
# return number of characters 3/20/21
def characterPrinter(characterList, save=False, returnImage=False):
    #f, axarr = plt.subplots(1, len(characterList))
    i = 0
    finalList = []
    for character in characterList:
        if (len(character) > 0):
            finalList.append(matrixToImg(character))
            #axarr[i].imshow(character, cmap="gray")
            i += 1
            if (save == True):
                matrixToImg(character).save(
                    "output" + str(i) + ".jpg")  # Eli Coltin 3/13/21

    # plt.show()
    # plt.close()
    if (returnImage == False):  # Eli Coltin 4/25/21
        return i  # Eli Coltin 3/20/21
    else:
        return finalList


# Eli Coltin
# 3/28/21
# Find the closest white pixel to a point with euclidian distance
# input 0: np.array : img_array [width , height , 3]
# input 1: grayscale image
# input 2: whether to rotate left or right, true is left, false is right
# input 3: amount of right side of image to look at when determining point to start searching for right angle at, default=.9
# output: 2D np.array: Yimg [width, height] cropped to this pixel in top left, grayscale, y level to start angle search at, x level
def closestPoint(img_array, grayscale, rotateDir, RHSearch=.9):
    q1 = 0
    q2 = 0

    if(rotateDir == False):
        q2 = len(img_array[0])

    # shortest distance, point storage
    shortestDist = np.Inf
    p1 = 0
    p2 = 0

    # iterate through each pixel
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            # if it is white pixel
            if img_array[i][j] == 255:
                # calc distance from point
                dist = (i-q1)**2+(j-q2)**2
                # if this is shortest dist
                if (dist < shortestDist):
                    # update storage values
                    shortestDist = dist
                    p1 = i
                    p2 = j

    if(rotateDir == False):
        # horizontal plot of edges, use to trim any leftover black space
        edges = edgeListCreator(threshGraph(img_array[p1:, 0:p2], 0))
        img_array = img_array[p1:, 0:edges[-1]]

        # with new edges, take the last 10% of the image
        # make row sum graph, look for the point where the last character approximately ends
        # get edge list of white points, look for about where the lowest row is - this is where to start the angle search
        yLevel = edgeListCreator(threshGraph(
            img_array[:, round(p2*RHSearch):], 1, 50))[-1]

        return(img_array, grayscale[p1:, 0:edges[-1]], yLevel, edges[-1])
    edges = edgeListCreator(threshGraph(img_array[p1:, p2:], 0))
    img_array = img_array[p1:, edges[0]:edges[-1]]
    edges2 = edgeListCreator(threshGraph(img_array, 1))
    return(img_array[edges2[0]:edges2[-1], :], grayscale[edges2[0]:edges2[-1], edges[0]:edges[-1]], 0, 0)


# Eli Coltin 3/14/21
# slopeInv will handle an inverse slope less than 1 and calculate how many white
# pixels are in the image at the angle
# if it is cleaning, then it will take the max slope and clean all the other pixels
# input:
#   2D np.array: inverse angle, image, y start value, x start value
# output:
#   2D np.array: dictionary of inverse slopes, img_array

def slopeInv(thisSlope, img_array, y, x):
    # amount of iterations
    amt = 0
    # amount of white pixels
    whitePixels = 0
    # flag to keep while loop going
    flag = True
    while(flag):
        amt += 1
        # make sure value is in the area of plate
        if(y+amt >= len(img_array) or round(x+amt*thisSlope) >= len(img_array[0]) or round(x+amt*thisSlope) < 0):
            break
        else:
            # if not cleaning and value in the area
            # if on the line, it is a white pixel
            if img_array[y+amt, round(x+amt*thisSlope)] > 0:
                # add to the white pixels
                whitePixels += 1
            # else:
                # use for visualizing line
                #img_array[y+amt, round(x+amt*thisSlope)] = 254
    # Negative direction
    amt = 0
    while(flag):
        amt += 1
        # make sure value is in the area of plate
        if(y-amt < 0 or round(x-amt*thisSlope) < 0 or round(x-amt*thisSlope) >= len(img_array[0])):
            break
        else:
            # if on the line, it is a white pixel
            if img_array[y-amt, round(x-amt*thisSlope)] > 0:
                # add to the white pixels
                whitePixels += 1
            # else:
                # use for visualizing line
                #img_array[y-amt, round(x-amt*thisSlope)] = 254

    return whitePixels, img_array


# Eli Coltin 3/14/21
# slopeNormal will handle an inverse slope greater than 1 and calculate how many white
# pixels are in the image at the angle
# if it is cleaning, then it will take the max slope and clean all the other pixels
# input:
#   2D np.array: inverse angle, image, y start value, x start value
# output:
#   2D np.array: dictionary of inverse slopes, img_array

def slopeNormal(thisSlope, img_array, y, x):
    # amount of iterations
    amt = 0
    # amount of white pixels
    whitePixels = 0
    # slope created from inverse slope
    thisSlope = 1/thisSlope
    # flag to keep while loop going
    flag = True

    # pos direction
    while(flag):
        amt += 1
        # make sure value is in the area of plate
        if(round(y+amt*thisSlope) >= len(img_array) or x+amt >= len(img_array[0]) or round(y+amt*thisSlope) < 0):
            break
        else:
            # if not cleaning and value in the area
            # if on the line, it is a white pixel
            if img_array[round(y+amt*thisSlope), x+amt] > 0:
                # add to the white pixels
                whitePixels += 1
            # use for visualizing line
            # else:
                #img_array[round(y+amt*thisSlope), x+amt] = 254

    # negative direction
    # set amount to 0
    amt = 0
    while(flag):
        amt += 1
        # make sure value is in the area of plate
        if(round(y-amt*thisSlope) < 0 or x-amt < 0 or round(y-amt*thisSlope) >= len(img_array)):
            break
        else:
            # if on the line, it is a white pixel
            if img_array[round(y-amt*thisSlope), x-amt] > 0:
                # add to the white pixels
                whitePixels += 1
            # else:
                # use for visualizing line
                #img_array[round(y-amt*thisSlope), x-amt] = 254

    return whitePixels, img_array


# Eli Coltin 3/14/21
# whiteCounter will count the different angle of line for each white pixel
# and count the one which goes through the most other white pixels
# input:
#   2D np.array: img_array, y start value, x start value
# output:
#   2D np.array: dictionary of inverse slopes
def whiteCounter(img_array, y, x):

    # define the distances for when using rho, theta hough transform
    #rhoRange = [i for i in range(0,maxImgLength)]
    # define the angles that will be checked
    #thetaRange = [i for i in range(-90,91,2)]

    # List of inverse angles
    invAngleList = []

    # upward sloping angles
    range0 = 1
    range1 = 201

    # if need downward sloping
    if(y != 0):
        range0 = -200
        range1 = 0

    # calculate the inverse angles in the degree range from 1 to 178
    for i in range(range0, range1):
        invAngleList.append(round(1/math.tan((i/10)*math.pi/180), 3))

    # index through pixels in image
    thisList = []

    # check each theta in checking range
    for angle in invAngleList:

        # count of white pixels
        whitePixels = 0
        # iterate through the y values specified
        # if the inverse slope is greater than 1, turn it into a regular slope
        # this is because it will allow the largest amount of pixels to be checked
        # in future steps
        # if(abs(angle)>=1):
        whitePixels, img_array = slopeNormal(angle, img_array, y, x)
        # don't need below for now
        # else:
        #whitePixels, img_array = slopeInv(angle, img_array, y, x)

        # if there are white pixels
        if(whitePixels != None):
            # append to counting list
            thisList.append(whitePixels)
        else:
            # if there are not white pixels, append 0
            thisList.append(0)

    # storage for values
    slopeMax = 0
    slopeVal = 0

    # find slope between each angle
    for i in range(0, 196, 5):  # Eli Coltin 4/18/21 count by 5, not 10
        thisSlope = (thisList[i + 4] - thisList[i]) / 5
        # one with largest slope will be the edge between characters and black space line
        if (thisSlope > slopeMax):
            slopeMax = thisSlope
            slopeVal = invAngleList[i]

    # calculate the angle to rotate the image by
    rotateAngle = math.atan(1/slopeVal)*180/math.pi

    # essentially no rotation, don't rotate
    if(round(rotateAngle, 4) == .1):
        rotateAngle = 0

    return rotateAngle


# Eli Coltin
# 3/28/21
# Determine which way the characters are rotated
# Argument 0: image of the just the rotated characters in the license plate
# Argument 1: amount of image to look at when determining rotation value, default .5
# Argument 2: The % of the top/bottom of the image that should be considered for the rotate, default .2
# Output: true for rotate left, false for rotate right
def rotateDirection(img_array, lookAmt=.5, topBotAmt=.2):
    # take the first half of the image
    halfImg = img_array[:, 0:round(len(img_array[0])*lookAmt)]

    # Logic: if needs to be rotated left, the earlier characters will be higher than the later characters
    # if it needs to be rotated right, the later characters will be higher than the earlier characters
    # So, top 20% should have more pixels if it needs to be rotated left

    # take mean of the white pixel sum of the first 20% rows
    topAmt = np.mean(np.sum(halfImg, 1)[0:round(len(halfImg)*topBotAmt)])

    # take mean of white pixel sum of the last 20% rows
    botAmt = np.mean(np.sum(halfImg, 1)[round(len(halfImg)*(1-topBotAmt)):])

    return(topAmt >= botAmt)


# Eli Coltin 3/1/21
# imageRotation algorithm rotates an image
# Argument 0: image path # Eli Coltin 4/25/21
# Argument 1: rotation amount
# Returns image rotated specific amount #4/25/21
def imageRotation(img_name, rotation):
    img = Image.open(img_name)
    img = img.convert('L')
    imgRot = img.rotate(0, fillcolor="black")
    matrix = arrayToMatrix(np.array(imgRot.getdata()), imgRot)

    # rotation modifier
    rotation = rotation * -1

    # angle in degrees
    angDeg = rotation * np.pi / 180

    # tangent of theta/2
    dTan = math.tan(angDeg / 2)  # Eli Coltin 4/11/21

    # create new blank matrix, fill blank spots with white
    # Eli Coltin 4/11/21
    blankMatrix = np.full((len(matrix), len(matrix[0])), fill_value=255)

    # iterate through each row of matrix
    for i in range(len(blankMatrix)):
        # iterate through each column
        for j in range(len(blankMatrix[0])):
            # Eli Coltin 4/11/21
            # determine which x,y value on old image will go where after rotation
            # shear 1
            xNew = round(j - i * dTan)
            yNew = i
            # shear 2
            xNew = xNew
            yNew = round(xNew * math.sin(angDeg) + yNew)
            # shear3
            xNew = round(xNew - yNew * dTan)
            yNew = yNew
            # if pixel will be within image boundary, transfer it rotated from old to new
            if (xNew < len(matrix[0]) and yNew < len(matrix) and xNew >= 0 and yNew >= 0):
                blankMatrix[int(yNew), int(xNew)] = matrix[i, j]

    return blankMatrix


# Eli Coltin 4/26/21
# imageRotation algorithm rotates an image
# Argument 0: matrix # Eli Coltin 4/26/21
# Argument 1: rotation amount
# Returns image rotated specific amount #4/25/21
def imageRotation_matrix(matrix, rotation):
    # rotation modifier
    rotation = rotation * -1

    # angle in degrees
    angDeg = rotation * np.pi / 180

    # tangent of theta/2
    dTan = math.tan(angDeg / 2)  # Eli Coltin 4/11/21

    # create new blank matrix, fill blank spots with white
    # Eli Coltin 4/11/21
    blankMatrix = np.full((len(matrix), len(matrix[0])), fill_value=255)

    # iterate through each row of matrix
    for i in range(len(blankMatrix)):
        # iterate through each column
        for j in range(len(blankMatrix[0])):
            # Eli Coltin 4/11/21
            # determine which x,y value on old image will go where after rotation
            # shear 1
            xNew = round(j - i * dTan)
            yNew = i
            # shear 2
            xNew = xNew
            yNew = round(xNew * math.sin(angDeg) + yNew)
            # shear3
            xNew = round(xNew - yNew * dTan)
            yNew = yNew
            # if pixel will be within image boundary, transfer it rotated from old to new
            if (xNew < len(matrix[0]) and yNew < len(matrix) and xNew >= 0 and yNew >= 0):
                blankMatrix[int(yNew), int(xNew)] = matrix[i, j]

    return blankMatrix


# Eli Coltin 4/25/21
# determine_image_rotation determines the image rotation
# argument 0 is the path
# returns the rotation amount
def determine_image_rotation(img_name):
    img = Image.open(img_name)
    img = img.convert('L')
    imgRot = img.rotate(0, fillcolor="white")
    imgRot = arrayToMatrix(np.array(imgRot.getdata()), imgRot)

    # create second, grayscale image which is needed to pass characters
    img2 = Image.open(img_name)
    img2 = img2.convert('L')
    imgGrayscale = img2.rotate(0, fillcolor="white")
    grayscale = arrayToMatrix(np.array(imgGrayscale.getdata()), img2)

    # median filter the image
    imgRot = median_filter(imgRot)

    # reverse threshold image
    imgRev = threshold(imgRot, otsu_threshold(imgRot), True)

    # Eli Coltin 4/25/21
    # if errors in this section, return none and only try segmenting
    try:
        # row graph of image
        horizGraph = threshGraph(imgRev, 1)
        # slice top and bottom off of image
        imgSlice = horizontalImageSlicer(verticalBoundaryFinder(imgRev, edgeListCreator(horizGraph)), imgRev, grayscale,
                                         .05)

        # column graph of image
        vertGraph = threshGraph(imgSlice[0], 0)

        # vertically slice edges of image
        imgHorizCrop = onlyEdgeSlicer(verticalBoundaryFinder(imgSlice[0], edgeListCreator(vertGraph)), imgSlice[0],
                                      imgSlice[1], True)

        # horizontal graph of just characters
        horizGraph2 = threshGraph(imgHorizCrop[0], 1)

        # slice image again to just characters
        imgVertCrop = onlyEdgeSlicer(verticalBoundaryFinder(imgHorizCrop[0], edgeListCreator(horizGraph2)),
                                     imgHorizCrop[0], imgHorizCrop[1], False)

        # crop image such that if it needs to be rotated left, left character in top left corner
        # if needs to be rotated right, right character in top right corner
        imgCloseCrop = closestPoint(
            imgVertCrop[0], imgVertCrop[1], rotateDirection(imgVertCrop[0]))

        # horizontal graph of just characters
        horizGraph3 = threshGraphDivisor(imgCloseCrop[0], 1, 5)

        if (len(verticalBoundaryFinder(imgCloseCrop[0], edgeListCreator(horizGraph3))) > 2):
            # slice image again to just characters
            imgVertCrop2 = onlyEdgeSlicer(verticalBoundaryFinder(imgCloseCrop[0], edgeListCreator(horizGraph3)),
                                          imgCloseCrop[0], imgCloseCrop[1], False)

            # crop image such that if it needs to be rotated left, left character in top left corner
            # if needs to be rotated right, right character in top right corner
            imgCloseCrop2 = closestPoint(
                imgVertCrop2[0], imgVertCrop2[1], rotateDirection(imgVertCrop2[0]))

            # set name back
            imgCloseCrop = imgCloseCrop2

        # get rotation amount
        rotAmt = whiteCounter(
            imgCloseCrop[0], imgCloseCrop[2], imgCloseCrop[3])
    except IndexError:
        return None

    return rotAmt


# Eli Coltin 4/25/21
# determine_image_rotation_matrix determines the image rotation for a matrix
# argument 0 is the matrix
# argument 1 is a copy of the matrix
# returns the rotation amount
def determine_image_rotation_matrix(matrix, grayscale):
    imgRot = matrix

    # median filter the image
    imgRot = median_filter(imgRot)

    # reverse threshold image
    imgRev = threshold(imgRot, otsu_threshold(imgRot), True)

    # Eli Coltin 4/25/21
    # if errors in this section, return none and only try segmenting
    try:
        # row graph of image
        horizGraph = threshGraph(imgRev, 1)
        # slice top and bottom off of image
        imgSlice = horizontalImageSlicer(verticalBoundaryFinder(imgRev, edgeListCreator(horizGraph)), imgRev, grayscale,
                                         .05)

        # column graph of image
        vertGraph = threshGraph(imgSlice[0], 0)

        # vertically slice edges of image
        imgHorizCrop = onlyEdgeSlicer(verticalBoundaryFinder(imgSlice[0], edgeListCreator(vertGraph)), imgSlice[0],
                                      imgSlice[1], True)

        # horizontal graph of just characters
        horizGraph2 = threshGraph(imgHorizCrop[0], 1)

        # slice image again to just characters
        imgVertCrop = onlyEdgeSlicer(verticalBoundaryFinder(imgHorizCrop[0], edgeListCreator(horizGraph2)),
                                     imgHorizCrop[0], imgHorizCrop[1], False)

        # crop image such that if it needs to be rotated left, left character in top left corner
        # if needs to be rotated right, right character in top right corner
        imgCloseCrop = closestPoint(
            imgVertCrop[0], imgVertCrop[1], rotateDirection(imgVertCrop[0]))

        # horizontal graph of just characters
        horizGraph3 = threshGraphDivisor(imgCloseCrop[0], 1, 5)

        if (len(verticalBoundaryFinder(imgCloseCrop[0], edgeListCreator(horizGraph3))) > 2):
            # slice image again to just characters
            imgVertCrop2 = onlyEdgeSlicer(verticalBoundaryFinder(imgCloseCrop[0], edgeListCreator(horizGraph3)),
                                          imgCloseCrop[0], imgCloseCrop[1], False)

            # crop image such that if it needs to be rotated left, left character in top left corner
            # if needs to be rotated right, right character in top right corner
            imgCloseCrop2 = closestPoint(
                imgVertCrop2[0], imgVertCrop2[1], rotateDirection(imgVertCrop2[0]))

            # set name back
            imgCloseCrop = imgCloseCrop2

        # get rotation amount
        rotAmt = whiteCounter(
            imgCloseCrop[0], imgCloseCrop[2], imgCloseCrop[3])
    except IndexError:
        return None

    return rotAmt


# Eli Coltin 4/25/21
# crop image down as necessary
# argument 1 is the image
# argument 2 is the grayscale version of the image
# returns a cropped version of the image and grayscale cropped version
def crop_image(image, grayscale):
    # median filter the image
    imgRot = median_filter(image)

    # width, length of image
    imgDim = [len(imgRot), len(imgRot[0])]

    # reverse threshold image
    imgRev = threshold(imgRot, otsu_threshold(imgRot), True)

    # row graph of image
    horizGraph = threshGraph(imgRev, 1)
    # slice top and bottom off of image
    imgSlice = horizontalImageSlicer(verticalBoundaryFinder(imgRev, edgeListCreator(horizGraph)), imgRev, grayscale,
                                     .05)

    # column graph of image
    vertGraph = threshGraph(imgSlice[0], 0)

    # vertically slice edges of image
    imgHorizCrop = onlyEdgeSlicer(verticalBoundaryFinder(imgSlice[0], edgeListCreator(vertGraph)), imgSlice[0],
                                  imgSlice[1], True)

    # horizontal graph of just characters
    horizGraph2 = threshGraph(imgHorizCrop[0], 1)

    if (len(onlyEdgeSlicer(verticalBoundaryFinder(imgHorizCrop[0], edgeListCreator(horizGraph2)), imgHorizCrop[0],
                           imgHorizCrop[1], False)[0]) > imgDim[0] * .1):
        # slice image again to just characters
        imgVertCrop = onlyEdgeSlicer(verticalBoundaryFinder(imgHorizCrop[0], edgeListCreator(horizGraph2)),
                                     imgHorizCrop[0], imgHorizCrop[1], False)
    else:
        imgVertCrop = imgHorizCrop

    return imgVertCrop[0], imgVertCrop[1]

# Eli Coltin 4/25/21
# input image name, rotate it, crop it, print out the characters in it
# argument 0: image path
# output: string of characters in the license plate


def character_segmenter(img_name):
    print("Starting character segmentation...")

    print("Starting image rotation...")
    # determine how much image must be rotated
    imgRotAmt = determine_image_rotation(img_name)

    # create two copies of the image, rotate both (if rotate is none, rotate 0)
    # if rotation is none, probably just image of characters, only segment the characters
    if (imgRotAmt != None):
        imgRot = imageRotation(img_name, imgRotAmt)
        grayscale = imageRotation(img_name, imgRotAmt)
    else:
        imgRot = imageRotation(img_name, 0)
        grayscale = imageRotation(img_name, 0)
    print("Image rotation completed")

    print("Starting image cropping...")
    # crop the image and copy if rotation is not none
    if (imgRotAmt != None):
        image, grayscale = crop_image(imgRot, grayscale)
    else:
        # if rotation is none, just change name, only segment
        # must first median filter, then otsu threshold
        imgMed = median_filter(imgRot)
        imgRev = threshold(imgMed, otsu_threshold(imgMed), True)
        image = imgRev

    # create pixel sum graph, by columns, to determine where characters are
    horizGraph = threshGraphDivisor(image, 0, 20)
    # create edges, find boundaries of characters, and determine the locations of each character
    characterList = characterListCreator(verticalBoundaryFinder(
        image, edgeListCreator(horizGraph)), image, grayscale)
    # trim characters horizontally and vertically, then print each found character
    trimmed = characterPrinter(characterTrimmer(characterList[0], characterList[1])[
                               1], save=True, returnImage=False)
    print("Image cropping completed")
    print("Character segmentation completed")
    # create string of predicted characters
    finalString = ""
    for char in range(int(trimmed)):
        st = "output" + str(char + 1) + ".jpg"
        finalString += neuralNetwork(st, "licence_1.h5")
    # for x in i[0]:
    # finalString += neuralNetwork(x, "licence_1.h5")

    #print(finalString)

    return finalString, trimmed


# Eli Coltin 4/25/21
# input image as matrix, rotate it, crop it, print out the characters in it
# argument 0: image path
# output: string of characters in the license plate
def character_segmenter_matrix(matrix):
    print("Starting character segmentation...")

    print("Starting image rotation...")
    # determine how much image must be rotated
    imgRotAmt = determine_image_rotation_matrix(matrix, matrix)

    # create two copies of the image, rotate both (if rotate is none, rotate 0)
    # if rotation is none, probably just image of characters, only segment the characters
    if (imgRotAmt != None):
        imgRot = imageRotation_matrix(matrix, imgRotAmt)
        grayscale = imageRotation_matrix(matrix, imgRotAmt)
    else:
        imgRot = imageRotation_matrix(matrix, 0)
        grayscale = imageRotation_matrix(matrix, 0)
    print("Image rotation completed")

    print("Starting image cropping...")
    # crop the image and copy if rotation is not none
    if (imgRotAmt != None):
        image, grayscale = crop_image(imgRot, grayscale)
    else:
        # if rotation is none, just change name, only segment
        # must first median filter, then otsu threshold
        imgMed = median_filter(imgRot)
        imgRev = threshold(imgMed, otsu_threshold(imgMed), True)
        image = imgRev

    # create pixel sum graph, by columns, to determine where characters are
    horizGraph = threshGraphDivisor(image, 0, 20)
    # create edges, find boundaries of characters, and determine the locations of each character
    characterList = characterListCreator(verticalBoundaryFinder(
        image, edgeListCreator(horizGraph)), image, grayscale)
    # trim characters horizontally and vertically, then print each found character
    trimmed = characterPrinter(characterTrimmer(characterList[0], characterList[1])[
                               1], save=True, returnImage=False)
    print("Image cropping completed")
    print("Character segmentation completed")
    # create string of predicted characters
    finalString = ""
    for char in range(int(trimmed)):
        st = "output" + str(char + 1) + ".jpg"
        finalString += neuralNetwork(st, "licence_1.h5")
    # for x in i[0]:
    # finalString += neuralNetwork(x, "licence_1.h5")

    #print(finalString)

    return finalString, trimmed