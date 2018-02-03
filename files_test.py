import os
import numpy as np
from PIL import Image

path2 = './leapmotion_dataset'
imlist = os.listdir(path2)
    
image1 = np.array(Image.open(path2 +'/' + imlist[0])) 

m,n = image1.shape[0:2] 
total_images = len(imlist) 

immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                        for images in imlist], dtype = 'f')



print (immatrix.shape)