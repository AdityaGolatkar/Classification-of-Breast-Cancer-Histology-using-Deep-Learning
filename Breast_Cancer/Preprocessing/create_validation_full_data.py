import shutil
import os
import random
import pdb
import numpy as np

def create_validation(source,dest,img_per_dir,val_per_dir):
    for d in range(len(source)):
        files = sorted(os.listdir(source[d]))
        #sel_index_complete = random.sample(range(img_per_dir),img_per_dir)
        #pdb.set_trace()
        #np.save('ordering.npy',sel_index_complete)
        sel_index_complete = np.load('ordering.npy')
        sel_index = sel_index_complete[50:75]
        sel_index.sort()
        #sel_index_temp = range(img_per_dir)
        #sel_index = sel_index_temp[25:50]
        j = 0
        i = 0
        for f in files:
            if j == sel_index[i]:
                shutil.move(source[d]+f,dest[d])
                i=i+1
            if i == len(sel_index):
                break
            j=j+1
            #print(j)

img_per_dir = 100
val_per_dir = 25

source = ['/home/Drive2/aditya/Breast_Cancer/Data/Training/Normal/',\
'/home/Drive2/aditya/Breast_Cancer/Data/Training/Benign/',\
'/home/Drive2/aditya/Breast_Cancer/Data/Training/InSitu/',\
'/home/Drive2/aditya/Breast_Cancer/Data/Training/Invasive/']

dest = ['/home/Drive2/aditya/Breast_Cancer/Data/Validation/Normal/',\
'/home/Drive2/aditya/Breast_Cancer/Data/Validation/Benign/',\
'/home/Drive2/aditya/Breast_Cancer/Data/Validation/InSitu/',\
'/home/Drive2/aditya/Breast_Cancer/Data/Validation/Invasive/']

create_validation(source,dest,img_per_dir,val_per_dir)
