if __name__ == "__main__":
    import sys
    from bc_detection import character_segmenter

    # get image path from argument
    img_name = str(sys.argv[1])
    plateString, num = character_segmenter(img_name)
    print(plateString)
    print(num)
    
    
    
