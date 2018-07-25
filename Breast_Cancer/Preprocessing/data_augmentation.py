from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import glob
import pdb
import os

######################################################################
no_samples = 25
benign_mapping_path = "/home/Drive2/aditya/benign_mapping.npy"
normal_mapping_path = "/home/Drive2/aditya/normal_mapping.npy"
insitu_mapping_path = "/home/Drive2/aditya/insitu_mapping.npy"
invasive_mapping_path = "/home/Drive2/aditya/invasive_mapping.npy"

benign_ss_path  = "/home/Drive2/aditya/benign_patches_info.npy"
normal_ss_path  = "/home/Drive2/aditya/normal_patches_info.npy"
insitu_ss_path  = "/home/Drive2/aditya/insitu_patches_info.npy"
invasive_ss_path  = "/home/Drive2/aditya/invasive_patches_info.npy"
#####################################################################

benign_subsamples = np.zeros((no_samples,1))
normal_subsamples = np.zeros((no_samples,1))
insitu_subsamples = np.zeros((no_samples,1))
invasive_subsamples = np.zeros((no_samples,1))
log = open("/home/Drive2/aditya/Mapping_file.txt","w")

def data_augmentation(input_image_path,image_name,path,img_dim):
    img = cv2.imread(input_image_path)
    threshold = 0.05
    img_shape = img.shape
    masked_img = mask_creator(img)
    img_area = img_dim*img_dim
    threshold_img_area = threshold*img_area
    area = []
    cords = []
    #worst_case_patch = np.zeros((299,299,3))
    #pdb.set_trace()
    #worst_case_patch = img[0:299,0:299,:]
    wc_patch_list = []
    #hvec = np.zeros((5,1))
    m = 0
    row_ind = 0
    while(True):
        col_ind = 0
        while(True):

            #if (np.sum(masked_img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim]) > area):
            #    area = np.sum(masked_img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim])
            #    worst_case_patch = img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim,:]
            area.append(np.sum(masked_img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim]))
            cords.append([row_ind,row_ind+img_dim,col_ind,col_ind+img_dim])

            if (np.sum(masked_img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim]) > threshold_img_area):
                subimage = img[row_ind:row_ind+img_dim,col_ind:col_ind+img_dim,:]
                cv2.imwrite(path+image_name+"_"+str(m)+".png", subimage)
                m = m + 1
                #print(m,1)
            col_ind = col_ind + img_dim//2
            if col_ind + img_dim > img_shape[1]:

                #if (np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]) > area):
                #     area = np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1])
                #     worst_case_patch = img[row_ind:row_ind+img_dim,-img_dim-1:-1,:]
                area.append(np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]))
                cords.append([row_ind,row_ind+img_dim,-img_dim-1,-1])
                
                if (np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]) > threshold_img_area):
                    subimage = img[row_ind:row_ind+img_dim,-img_dim-1:-1,:]
                    cv2.imwrite(path+image_name+"_"+str(m)+".png", subimage)
                    m = m + 1
                    #print(m,2)
                break
        row_ind = row_ind + img_dim//2
        if row_ind + img_dim  > img_shape[0]:
            break

    col_ind = 0
    while(True):

        #if (np.sum(masked_img[-img_dim-1:-1,col_ind:col_ind+img_dim]) > area):
        #    area = np.sum(masked_img[-img_dim-1:-1,col_ind:col_ind+img_dim])
        #    worst_case_patch = img[-img_dim-1:-1,col_ind:col_ind+img_dim,:]

        area.append(np.sum(masked_img[row_ind:row_ind+img_dim,-img_dim-1:-1]))
        cords.append([-img_dim-1,-1,col_ind,col_ind+img_dim])

        if (np.sum(masked_img[-img_dim-1:-1,col_ind:col_ind+img_dim]) > threshold_img_area):
            subimage = img[-img_dim-1:-1,col_ind:col_ind+img_dim,:]
            cv2.imwrite(path+image_name+"_"+str(m)+".png", subimage)
            m = m + 1
            #print(m,3)
        col_ind = col_ind + img_dim//2
        if col_ind + img_dim > img_shape[1]:
            break

    if m < 50:
        no_img_needed = 50-m
        order = sorted(range(len(area)), key=lambda k: area[k], reverse=True)
        
        for i in range(no_img_needed):
            subimage = img[cords[order[i]][0]:cords[order[i]][1],cords[order[i]][2]:cords[order[i]][3]]
            cv2.imwrite(path+image_name+"_"+str(m)+".png", subimage)
            m = m + 1

        #cv2.imwrite(path+image_name+"_"+str(m)+".png", worst_case_patch)
        #m = m + 1
        

    return m        

def mask_creator(img):
    threshold = 0.6
    mask_img = np.zeros((img.shape[0],img.shape[1]))
    r_channel = img[:,:,0]
    b_channel = img[:,:,2]
    r_channel[(r_channel==0)] = -1
    b_by_r = b_channel/r_channel
    mask_img[(b_by_r<threshold)] = 1
    return mask_img


def image_reader():
    dirlist=['/home/Drive2/aditya/Data/Photos/Training/Benign/*png',
	'/home/Drive2/aditya/Data/Photos/Training/Normal/*png',
        '/home/Drive2/aditya/Data/Photos/Training/InSitu/*png',    
        '/home/Drive2/aditya/Data/Photos/Training/Invasive/*png',
        '/home/Drive2/aditya/Data/Photos/Validation/Benign/*png',
	'/home/Drive2/aditya/Data/Photos/Validation/Normal/*png',
        '/home/Drive2/aditya/Data/Photos/Validation/InSitu/*png',    
        '/home/Drive2/aditya/Data/Photos/Validation/Invasive/*png']
 
    print(len(dirlist),file=log)
    
    benign_mapping = np.zeros((no_samples,1))
    insitu_mapping = np.zeros((no_samples,1))
    invasive_mapping = np.zeros((no_samples,1))
    normal_mapping = np.zeros((no_samples,1))
    
    for m in range(len(dirlist)):
	ct = 0
	for filename in sorted(glob.glob(dirlist[m])):
	    image_name = filename[int(len(filename)-filename[::-1].find("/")):filename.find(".")]
	    img_no = int(image_name[-3:])
            #print("Img_NO ",img_no)
            if m == 0:
		path = '/home/Drive2/aditya/Aug_Data/Training/Benign/'
	    elif m == 1:
		path = '/home/Drive2/aditya/Aug_Data/Training/Normal/'
	    elif m == 2:
		path = '/home/Drive2/aditya/Aug_Data/Training/InSitu/'
	    elif m == 3:
	        path = '/home/Drive2/aditya/Aug_Data/Training/Invasive/'
	    elif m == 4:
	        path = '/home/Drive2/aditya/Aug_Data/Validation/Benign/'
	        benign_mapping[ct] = img_no
            elif m == 5:
		path = '/home/Drive2/aditya/Aug_Data/Validation/Normal/'
                normal_mapping[ct] = img_no
	    elif m == 6:
		path = '/home/Drive2/aditya/Aug_Data/Validation/InSitu/'
                insitu_mapping[ct] = img_no
	    elif m == 7:
		path = '/home/Drive2/aditya/Aug_Data/Validation/Invasive/'
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
