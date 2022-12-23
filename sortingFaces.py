import os
import pandas as pd


df = pd.read_csv('./SOURCE_FOLDER_FOR_IMAGES/list_attr_celeba.csv')
smiles = df["Smiling"].replace(-1, 0).tolist()

src = 'SOURCE_FOLDER_FOR_IMAGES'
smiling = 'SOURCE_FOLDER_FOR_NEW_FOLDER/smiling/'
nosmile = 'SOURCE_FOLDER_FOR_NEW_FOLDER/nosmile/'

for i in range(len(smiles)):
    filename = "{:06d}".format(i+1) + ".jpg"
    if(smiles[i] == 0):
        os.replace(src + filename, nosmile + filename)
    else:
        os.replace(src + filename, smiling + filename)