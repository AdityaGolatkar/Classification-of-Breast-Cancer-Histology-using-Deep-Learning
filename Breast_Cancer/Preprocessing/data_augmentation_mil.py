from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import glob
import pdb
import os

##############
# Parameters #
##############
patch_size = 299
minimal_patches = 20
overlap = 0.25
area_threshold = 0.02
mask_threshold = 0.63

#########
# Paths #
#########
norm_full_path = '/home/Drive2/aditya/MIL/Data_Norm'
norm_full_type = '*png'
norm_patches_path = '/home/Drive2/aditya/MIL/Data_Norm_Patches'
mask_path = '/home/Drive2/aditya/MIL/Masks/'
log = open("/home/Drive2/aditya/Mapping_file.txt","w")


print("0=Benign,1=Malinant",file=log)

def data_augmentation(input_image_path,image_name,path,img_dim):
    
    img = cv2.imread(input_image_path)
    threshold = area_threshold
    img_shape = img.shape
    masked_img = mask_creator(img,image_name)

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

            col_ind = col_ind + int((1-overlap)*img_dim)

            if col_ind + img_dim > img_shape[1]:

                area.append(np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]))
                cords.append([row_ind,row_ind+img_dim,-img_dim-1,-1])
                
                if (np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]) > threshold_img_area):
                    m = m + 1
                break

        row_ind = row_ind + int((1-overlap)*img_dim)

        if row_ind + img_dim  > img_shape[0]:
            break

    col_ind = 0
    while(True):

        area.append(np.sum(masked_img[-img_dim-1:-1,col_ind:col_ind+img_dim]))
        cords.append([-img_dim-1,-1,col_ind,col_ind+img_dim])

        if (np.sum(masked_img[-img_dim-1:-1,col_ind:col_ind+img_dim]) > threshold_img_area):
            m = m + 1
        
        col_ind = col_ind + int((1-overlap)*img_dim)
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

    if no_img_needed < minimal_patches:
        no_img_needed = minimal_patches

    order = sorted(range(len(area)), key=lambda k: area[k], reverse=True)
        
    for i in range(no_img_needed):
        subimage = img[cords[order[i]][0]:cords[order[i]][1],cords[order[i]][2]:cords[order[i]][3]]
        cv2.imwrite(path+image_name+"_"+str(i)+".png", subimage)
    
    return no_img_needed  


def mask_creator(img):
    threshold = mask_threshold
    mask_img = np.zeros((img.shape[0],img.shape[1]))
    r_channel = img[:,:,0]
    b_channel = img[:,:,2]
    r_channel[(r_channel==0)] = -1
    b_by_r = b_channel/r_channel
    mask_img[(b_by_r<threshold)] = 1
    cv2.imwrite(mask_path+image_name,mask_img)
    return mask_img


def image_reader():
    dirlist=[norm_full_path+'/Training/Training/Benign/'+norm_full_type,
	norm_full_path+'/Training/Training/Malignant/'+norm_full_type,
        norm_full_path+'/Training/Validation/Benign/'+norm_full_type,
        norm_full_path+'/Training/Validation/Malignant/'+norm_full_type,
        norm_full_path+'/Testing/Benign/'+norm_full_type,
        norm_full_path+'/Testing/Malignant/'+norm_full_type]
 
    print(len(dirlist))
     
    for m in range(len(dirlist)):
	ct = 0
	for filename in sorted(glob.glob(dirlist[m])):
	    image_name = filename[int(len(filename)-filename[::-1].find("/")):filename.find(".")]
	    img_no = int(image_name[-3:])
            path = dirlist[m].replace('Data_Norm','Data_Norm_Patches')[0:-len(norm_full_type]
            #print("Img_NO ",img_no) 
	    patches = data_augmentation(filename,image_name,path,patch_size)
            if patches == 0:
                print("Change the  threshold")	    
           
            print("PCIm",patches,ct,image_name,m,file=log)
            print("PCIm",patches,ct,image_name,m)
            ct = ct+1
	    #print("COunt ",ct)

image_reader()
