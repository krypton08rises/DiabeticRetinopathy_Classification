import os
import cv2
import numpy as np

path = "./Datasets/Kaggle/Train"
# cPath = "./Datasets/Kaggle/compressedTrain"

for file in os.listdir(path):
    inp =path + '/'+file
    out = cPath+ '/'+file
    img = cv2.imread(inp)
    img = cv2.resize(img, (600, 400), cv2.INTER_AREA)
    cv2.imwrite(out,img)
