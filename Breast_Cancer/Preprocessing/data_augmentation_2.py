from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import glob
import pdb
import os

######################################################################
no_samples = 25
benign_mapping_path = "/home/Drive2/aditya/fold11/benign_mapping.npy"
normal_mapping_path = "/home/Drive2/aditya/fold11/normal_mapping.npy"
insitu_mapping_path = "/home/Drive2/aditya/fold11/insitu_mapping.npy"
invasive_mapping_path = "/home/Drive2/aditya/fold11/invasive_mapping.npy"

benign_ss_path  = "/home/Drive2/aditya/fold11/benign_patches_info.npy"
normal_ss_path  = "/home/Drive2/aditya/fold11/normal_patches_info.npy"
insitu_ss_path  = "/home/Drive2/aditya/fold11/insitu_patches_info.npy"
invasive_ss_path  = "/home/Drive2/aditya/fold11/invasive_patches_info.npy"
#####################################################################

benign_subsamples = np.zeros((no_samples,1))
normal_subsamples = np.zeros((no_samples,1))
insitu_subsamples = np.zeros((no_samples,1))
invasive_subsamples = np.zeros((no_samples,1))
log = open("/home/Drive2/aditya/fold11/Mapping_file.txt","w")

print("0=benign,1=normal,2=insitu,3=invasive",file=log)

def data_augmentation(input_image_path,image_name,path,img_dim):
    
    img = cv2.imread(input_image_path)
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


def image_reader():
    dirlist=['/home/Drive2/aditya/Data1/Photos/Training/Benign/*tif',
	'/home/Drive2/aditya/Data1/Photos/Training/Normal/*tif',
        '/home/Drive2/aditya/Data1/Photos/Training/InSitu/*tif',    
        '/home/Drive2/aditya/Data1/Photos/Training/Invasive/*tif',
        '/home/Drive2/aditya/Data1/Photos/Validation/Benign/*tif',
	'/home/Drive2/aditya/Data1/Photos/Validation/Normal/*tif',
        '/home/Drive2/aditya/Data1/Photos/Validation/InSitu/*tif',    
        '/home/Drive2/aditya/Data1/Photos/Validation/Invasive/*tif']
 
    print(len(dirlist))
    
    benign_mapping = np.zeros((no_samples,1))
    insitu_mapping = np.zeros((no_samples,1))
    invasive_mapping = np.zeros((no_samples,1))
    normal_mapping = np.zeros((no_samples,1))
    
    for m in range(len(dirlist)):
	if m//2 == 0:
            ct = 0
	for filename in sorted(glob.glob(dirlist[m])):
	    image_name = filename[int(len(filename)-filename[::-1].find("/")):filename.find(".")]
	    img_no = int(image_name[-3:])
            #print("Img_NO ",img_no)
            if m == 0 or m == 1:
		path = '/home/Drive2/aditya/Aug_Data/Train2/NonCancer/'
	    elif m == 2 or m == 3:
		path = '/home/Drive2/aditya/Aug_Data/Train2/Cancer/'
	    elif m == 4 or m == 5:
	        path = '/home/Drive2/aditya/Aug_Data/Val2/NonCancer/'
	        benign_mapping[ct] = img_no
            elif m == 5:
		path = '/home/Drive2/aditya/Aug_Data1/Validation/Normal/'
                normal_mapping[ct] = img_no
	    elif m == 6:
		path = '/home/Drive2/aditya/Aug_Data1/Validation/InSitu/'
                insitu_mapping[ct] = img_no
	    elif m == 7:
		path = '/home/Drive2/aditya/Aug_Data1/Validation/Invasive/'
                invasive_mapping[ct] = img_no

	    patches = data_augmentation(filename,str(ct),path,299)
            if patches == 0:
                print("Change the  threshold")	    
            if m == 4:
                benign_subsamples[ct] = patches
            elif m == 5:
                normal_subsamples[ct] = patches
            elif m == 6:
                insitu_subsamples[ct] = patches
            elif m == 7:
                invasive_subsamples[ct] = patches
            print("PCIm",patches,ct,img_no,m,file=log)
            print("PCIm",patches,ct,img_no,m)
            ct = ct+1
	    #print("COunt ",ct)

    np.save(benign_mapping_path,benign_mapping)
    np.save(normal_mapping_path,normal_mapping)
    np.save(insitu_mapping_path,insitu_mapping)
    np.save(invasive_mapping_path,invasive_mapping)

    np.save(benign_ss_path,benign_subsamples)
    np.save(normal_ss_path,normal_subsamples)
    np.save(insitu_ss_path,insitu_subsamples)
    np.save(invasive_ss_path,invasive_subsamples)

image_reader()
