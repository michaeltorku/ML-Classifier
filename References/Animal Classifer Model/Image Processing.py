import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pandas as pd
from PIL import Image

X = 0
image_list = []
for filename in glob.glob(r'\Users\amoah_k\Desktop\Extended Essay\Extended Essay Dataset\Dataset\CNN_Data\training_set\cats/*.jpg'): #Picks from images from given file location
    im = Image.open(filename)
    im = im.resize((64,64), Image.ANTIALIAS)
    image_list.append(im)
    

df = pd.DataFrame()

n = 0

def MakeStoreVector():
    n = 0
    for X in image_list:
        n = n + 1
        X = X.convert('L') #Changes to grayscale
        
        matrix = np.asarray(X.getdata(), dtype = np.float64).reshape((X.size[1], X.size[0]))#Stores every pixel as a matrix
        
        vector = np.asarray(matrix).reshape(-1) #Changes matrix into a 1 Dimensional Vector
        
        print(n)
        
        df["0" * n] = vector 
        df.add
        
        if n == 500: #THIS n value lets you pick the number of images you extract
            df.to_csv(r'\Users\amoah_k\Desktop\Extended Essay\Extended Essay Dataset\Dataset\CNN_Data\training_set\cats/Dataset2.csv') # Where you want the excel file to go
            break

MakeStoreVector()
print("DONE")

#
