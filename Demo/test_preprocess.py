from __future__ import division
import numpy as np
import os
import cv2
import shutil

#Please provide the image name using the complete path.Even though the code and the image are in the same
#directory please provide the entire path. For ex: 

def test_pre_process(img_path):
    try:
        shutil.rmtree("Test_img_folder/")
    except:
        x=1
    os.makedirs("Test_img_folder/subfolder/")
    last_slash = img_path[::-1].find("/")
    img_name = img_path[int(len(img_path)-last_slash):img_path.find(".")]
    img = cv2.imread(img_path)
    no_of_patches = data_augmentation(img,img_name,"Test_img_folder/subfolder/",299)
    return no_of_patches, img_name

def data_augmentation(img,image_name,path,img_dim):
    
    threshold = 0.02
    img_shape = img.shape
    masked_img = mask_creator(img)

    ##########################################################
    metric = ((masked_img==1).sum()/np.prod(masked_img.shape))
    ##########################################################

    img_area = img_dim*img_dim
    threshold_img_area = threshold*img_area
    area = []
    cords = []
      
    m = 0
    row_ind = 0
    while(True):
        col_ind = 0
        while(True):

            area.append(np.sum(masked_img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim]))
            cords.append([row_ind,row_ind+img_dim,col_ind,col_ind+img_dim])

            if (np.sum(masked_img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim]) > threshold_img_area):
                m = m + 1

            col_ind = col_ind + img_dim//2

            if col_ind + img_dim > img_shape[1]:

                area.append(np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]))
                cords.append([row_ind,row_ind+img_dim,-img_dim-1,-1])
                
                if (np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]) > threshold_img_area):
                    m = m + 1
                break

        row_ind = row_ind + img_dim//2

        if row_ind + img_dim  > img_shape[0]:
            break

    col_ind = 0
    while(True):

        area.append(np.sum(masked_img[-img_dim-1:-1,col_ind:col_ind+img_dim]))
        cords.append([-img_dim-1,-1,col_ind,col_ind+img_dim])

        if (np.sum(masked_img[-img_dim-1:-1,col_ind:col_ind+img_dim]) > threshold_img_area):
            m = m + 1
        
        col_ind = col_ind + img_dim//2
        if col_ind + img_dim > img_shape[1]:
            break
    
    
    if metric > 0.01:
        no_img_needed = m
    elif metric < 0.01 and metric > 0.005:
        no_img_needed = 10
    elif metric < 0.005 and metric > 0.001:
        no_img_needed = 5
    else:
        no_img_needed = 1

    order = sorted(range(len(area)), key=lambda k: area[k], reverse=True)
        
    for i in range(no_img_needed):
        subimage = img[cords[order[i]][0]:cords[order[i]][1],cords[order[i]][2]:cords[order[i]][3]]
        cv2.imwrite(path+image_name+"_"+str(i)+".png", subimage)
    
    return no_img_needed  


def mask_creator(img):
    threshold = 0.63
    mask_img = np.zeros((img.shape[0],img.shape[1]))
    r_channel = img[:,:,0]
    b_channel = img[:,:,2]
    r_channel[(r_channel==0)] = -1
    b_by_r = b_channel/r_channel
    mask_img[(b_by_r<threshold)] = 1
    return mask_img
    

