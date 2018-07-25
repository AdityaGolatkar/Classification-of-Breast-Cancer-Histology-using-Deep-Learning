
# coding: utf-8

# In[2]:


import numpy as np
import cv2
#import spams
import numpy.matlib
import sklearn
from sklearn.decomposition import NMF
from sklearn import linear_model
#get_ipython().run_line_magic('matplotlib', 'inline')
import spams
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.misc
import glob
#import openslide


# In[3]:


def definePar(nstains,Lambda,batch):
    param = {'K' : nstains, # learns a dictionary with 100 elements
          'lambda1' : Lambda,
          'mode' : 2,
          'posAlpha' :True,
          'posD' :True,
          'modeD' : 0,
          #'lambda1' : 0.05,
          'numThreads' : -1,
          'batchsize':np.int(batch),
          'iter' : 200,#'verbose':False,
          #'whiten':0,
          #'clean':1,
             
          }
    return param


# In[4]:


def vectorize(array_np):
    array_np=np.reshape(array_np,array_np.size)
    return array_np


# In[5]:


#source=cv2.imread('source1.png')
def BLtrans(image):
	#b,g,r = cv2.split(bgr_img)
	#image=cv2.merge([r,g,b])
	#image is a numpy array
	Ivecd=np.float32(np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2])))
	V=np.log(255)-np.log(Ivecd+1);#V=WH, +1 is to avoid divide by zero
	#exclusion of white pixels
	out=cv2.cvtColor(image, cv2.COLOR_RGB2LAB)#matched on D65 whitepixel refence
	#print('out shape',out.shape)
	luminlayer=out[:,:,0]
	#print('luminal',luminlayer.shape)
	luminlayer= np.reshape(luminlayer,luminlayer.size)
	indices=np.where(np.float32(luminlayer)/255.0 <0.9)
	indices=np.squeeze(indices,axis=2)
	Inew=Ivecd[indices,:]
	#print('Inew',Inew.shape)
	VforW=np.log(255)-np.log(Inew+1)
	#VforW=np.squeeze(VforW, axis=0)
	#print('Vfor',VforW.shape)
	return V,VforW
#V,VforW=BLtrans(source)
#print np.where(VforW<0)
#print V.shape
#print VforW.shape


# In[6]:


def get_staincolor_sparsenmf(v,param):
	#print(param,v.T.shape)
	#print np.where(v.T<0)
	#parame = { 'K' :2,'lambda1': 0.02,'numThreads' :4,'posAlpha':True}
	Ws = spams.trainDL(v.T,**param)
	#print('here')    
	#model = NMF(n_components=param['K'], init='random', random_state=2,alpha=0.02, l1_ratio=1)
	##print(len(v<0))
	#Ws = model.fit_transform(v, y=None, W=None, H=None)
	#print("W_stain shape",Ws.shape)
	#print Ws.shape
	Ws=Ws[:,np.argsort(-Ws[1,:])]
	return Ws


# In[7]:


def estH(v,Ws, param,nrows,ncols):
	#print(Ws.shape,v.T.shape)
	param['pos']=True
	del param['posD']
#	del param['K']
	del param['batchsize']
	del param['modeD']
	del param['posAlpha']
	del param['iter']
	Hs_vec=spams.lasso(v.T,np.asfortranarray(Ws),return_reg_path = False,mode=2,lambda1=param['lambda1'],lambda2=0)
	Hs_vec=Hs_vec.toarray()
	#print(Hs_vec.shape)
	#model = linear_model.Lasso(alpha=0.02,positive=True)
	#print("W_stain shape",Ws.shape)
	#print("V shape",v.shape)
	#model.fit(v,Ws)
	#Hs_vec=(model.coef_)
	#print("Hs_stain vec shape",Hs_vec.shape)
	Hs = np.reshape(Hs_vec.T, (nrows, ncols, param['K']))
	#print("Hs_stain  shape",Hs.shape)
	iHs=[]#np.zeros((nrows,ncols,6),dtype=np.uint8)
	vdas=[]
	Irecon=[]
	for i in range(param['K']):
		vdAS =  np.multiply(np.expand_dims(Hs_vec.T[:,i],axis=1),np.expand_dims(Ws.T[i,:],axis=0))
		#print "Done"
		iHs.append(np.uint8(255.0*np.reshape(np.exp(-vdAS), (nrows,ncols,3))))
	Irecon = np.dot(Ws,Hs_vec)
	Irecon = 255*np.reshape(np.exp(-Irecon), (nrows, ncols, 3))
	Irecon = np.uint8(Irecon)
	return Hs,iHs,Irecon


# In[8]:


def stainstep(image,nstains,Lambda):
	V,V1=BLtrans(image)
	#print("Done BL")
	param=definePar(nstains,Lambda,round(0.2*V1.shape[0]))
	#print("Done Param")    
	Wi=get_staincolor_sparsenmf(V1,param)
	#print("Done SSNMF")    
	Hi,sepstains,_=estH(V,Wi,param,image.shape[0],image.shape[1])
	Hiv=vectorize(Hi)
	return Wi, Hi,Hiv,sepstains


# In[9]:


def SCN(source,Hta,Wta,Hso):
	Hso=np.reshape(Hso,(Hso.shape[0]*Hso.shape[1],Hso.shape[2]))
	#print("Hso shape" ,Hso.shape)
	Hso_Rmax = np.percentile(Hso,99,0,interpolation='midpoint') # 95 precentile of values in each column
	#print("Hso Rmax shape",Hso_Rmax.shape)
	Hta=np.reshape(Hta,(Hta.shape[0]*Hta.shape[1],Hta.shape[2]))
	##print("Hta shape",Hta.shape)
	Hta_Rmax = np.percentile(Hta,99,0,interpolation='midpoint')
	#print("Hta Rmax shape",Hta_Rmax.shape)
	normfac=Hta_Rmax/Hso_Rmax
	#print("normfac shape",normfac.shape)
	#m=numpy.matlib.repmat(normfac,Hso.shape[0],1)
	##print np.unique(m)
	#print Wta.shape
	Hsonorm = numpy.multiply(Hso,numpy.matlib.repmat(normfac,Hso.shape[0],1))
	#print("hs norm",Hsonorm.shape)
	Ihat=numpy.dot(Wta,Hsonorm.T)
	#print Ihat.shape
	sourcenorm=np.uint8(255.0*np.exp(-np.reshape(Ihat.T,(source.shape[0],source.shape[1],source.shape[2]))))
	return sourcenorm


# In[10]:


def color_normalization(source_path,Wi, Hi,nstains,save_path,Lambda):
    source=cv2.imread(source_path)
    b,g,r = cv2.split(source)
    source=cv2.merge([r,g,b])
    #target=cv2.imread(target_path)
    #b1,g1,r1 = cv2.split(target)
    #target=cv2.merge([r1,g1,b1])
    nstains=nstains;
    Lambda=Lambda;  # Use smaller values of the lambda (0.01-0.1) for better reconstruction. however, if the normalized image seems not fine, increase or decrease the value accordingly.

    # %% Our Method (The techniques is published in ISBI 2015 under the title "STRUCTURE-PRESERVED COLOR NORMALIZATION FOR HISTOLOGICAL IMAGES")
    # % For queries, contact: abhishek.vahadane@gmail.com, vahadane@iitg.ernet.in
    # % Source and target stain separation and storage of factors 
    # tic

    Wis, His,Hivs,_=stainstep(source,nstains,Lambda)
    #% save('source.mat','Wis','His','Hivs')
    #Wi, Hi,Hiv,_=stainstep(target,nstains,Lambda)
    our=SCN(source,Hi,Wi,His)
    #print("wis shape", Wis.shape)
    #print("HIs shape", His.shape)
    #print("Hivs shape", Hivs.shape)
    #print("Wi shape", Wi.shape)
    #print("Hi shape", Hi.shape)
    #print("Hiv shape", Hiv.shape)
    #plt.figure();plt.imshow(source)
    #plt.figure();plt.imshow(target)
    #plt.figure();plt.imshow(our)
    b,g,r = cv2.split(our)
    our=cv2.merge([r,g,b])
    cv2.imwrite(save_path, our)
    return our


# In[12]:


count000=0
target=cv2.imread('/home/aditya/Desktop/Breast_Cancer/Preprocessing/TCGA-E2-A14V-01Z-00-DX1.tif')
print(target.shape)
b1,g1,r1 = cv2.split(target)
target=cv2.merge([r1,g1,b1])
Wi, Hi,Hiv,_=stainstep(target,2,0.02)
#f_in=open('input.txt')

filename = "/home/aditya/Desktop/Breast_Cancer/Preprocessing/cs_img_1.png"
normalised = color_normalization(filename,Wi, Hi,2,"/home/aditya/Desktop/Breast_Cancer/Preprocessing/cn_img.png",0.02)



