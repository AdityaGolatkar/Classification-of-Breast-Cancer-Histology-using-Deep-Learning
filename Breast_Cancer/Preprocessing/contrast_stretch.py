from __future__ import division
import numpy as np
import cv2
import pdb


def contrast_stretch(img_path):
    img = cv2.imread(img_path)
    out = np.zeros(img.shape)
    b_channel = img[:,:,0]
    g_channel = img[:,:,1]
    r_channel = img[:,:,2]
    #pdb.set_trace() 
    #out[0:,0:,0] = (b_channel-np.min(b_channel))*(255/(np.max(b_channel)-np.min(b_channel)))
    #out[0:,0:,1] = (g_channel-np.min(g_channel))*(255/(np.max(g_channel)-np.min(g_channel)))
    #out[0:,0:,2] = (r_channel-np.min(r_channel))*(255/(np.max(r_channel)-np.min(r_channel))) 
    #out[:,:,0] = cv2.equalizeHist(b_channel)
    out[:,:,0] = b_channel
    out[:,:,1] = cv2.equalizeHist(g_channel)
    #out[:,:,1] = g_channel 
    #out[:,:,2] = cv2.equalizeHist(r_channel)
    out[:,:,2] = r_channel
    cv2.imwrite("cs_img_cs.png",out)

contrast_stretch("/home/Drive2/aditya/Data/Photos/Training/Benign/b032.tif")
