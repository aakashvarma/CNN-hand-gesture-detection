import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# path = '.\dataset'
filename = 'iiiok2.png'

# for filename in os.listdir(path):

img = cv2.imread(filename,1)
arr = np.array(img)
cv2.imshow('image',img)
k = cv2.waitKey(0)
print (filename)
if k == 27:       
    cv2.destroyAllWindows()

