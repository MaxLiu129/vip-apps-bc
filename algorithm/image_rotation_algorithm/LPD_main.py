if __name__ == "__main__":
    import sys
    from segmentation_lib import character_segmenter

    # get image path from argument
    img_name = str(sys.argv[1])
    plateString, num = character_segmenter(img_name)
    print(plateString)
    print(num)


# Eli Coltin 4/26/21
# Can be called, will segment license plate
def character_segmentation(matrix):
    from segmentation_lib import character_segmenter_matrix
    plateStr, num = character_segmenter_matrix(matrix)
    print(plateStr)
    print(num)
    return plateStr