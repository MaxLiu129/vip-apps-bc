from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    
    data = np.asarray( img, dtype="int32" )
    # output the data into a txt file to see the matrix before the change
    raw_file = open("test3_raw.txt","w")
    for row in data:
        np.savetxt(raw_file, row.astype(int)) 
        raw_file.write("---------------------------------------------------------------------------------------------------\n")   
    raw_file.close()
    
    # modify the matrix numbers
    data = (data[:,:,:3] * [0.2126, 0.7152, 0.0722]).sum(axis=2)

    print(data)
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )


data = load_image("test3.png")
save_image(data , "result3_709.png")