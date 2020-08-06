import pandas as pd
import numpy as np
import shutil,os

path_to_imgs = './Datasets/Kaggle/compressedTrain/'
finalfolder = './Datasets/Kaggle/prol_severe/'
path = './Datasets/Kaggle/trainLabels.csv'

df = pd.read_csv(path)

i=0
imgs = []           #images to add in the proliferate and sever categories
arr = []

for index, row in df.iterrows():
    if row['level'] == 3 or row['level'] == 4:
        arr.append([row['image'],row['level']-3])
        img = os.path.join(path_to_imgs, row['image']+'.jpeg')
        imgs.append(img)
        i+=1

arr = np.array(arr)

for file in imgs:
    shutil.copy(file,finalfolder)

new_df = pd.DataFrame(data = arr, columns = ['image', 'level'])
new_df.to_csv('./Datasets/Kaggle/newtrainlabels.csv')
