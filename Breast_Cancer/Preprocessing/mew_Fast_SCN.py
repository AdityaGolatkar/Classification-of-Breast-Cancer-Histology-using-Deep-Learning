import spams
import numpy as np
from scipy import sparse
from numpy import linalg
import matplotlib.pyplot as plt
import time
import math
import sys
from skimage import color
import os
import glob
import cv2
from sklearn import preprocessing
import multiprocessing
from multiprocessing import Pool
from functools import partial
from contextlib import closing
import openslide
from PIL import Image
import signal
import warnings
from tensorflow.python.framework import ops
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import sys
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

#to use cpu instead of gpu, uncomment the below line
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #use only GPU-1
os.environ['CUDA_VISIBLE_DEVICES'] = '' #use only CPU

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)
# config = tf.ConfigProto(device_count={'GPU': 1},log_device_placement=False,gpu_options=gpu_options)
config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)



# Please cite below paper if you use this code: 

# @inproceedings{Vahadane2015ISBI,
# 	Author = {Abhishek Vahadane and Tingying Peng and Shadi Albarqouni and Maximilian Baust and Katja Steiger and Anna Melissa Schlitter and Amit Sethi and Irene Esposito and Nassir Navab},
# 	Booktitle = {IEEE International Symposium on Biomedical Imaging},
# 	Date-Modified = {2015-01-31 17:49:35 +0000},
# 	Title = {Structure-Preserved Color Normalization for Histological Images},
# 	Year = {2015}}

# Contact: vahadane@iitg.ernet.in, abhishek.vahadane@gmail.com
# input image should be color image
# Python implementation by: Goutham Ramakrishnan, goutham7r@gmail.com



def main():
	op=2
	# 1= stain separation demo 
	# 2= color normalization of one image with one target image
	# 3= color normalization of all images in a folder with one target image
	# 4= color normalization of one image with multiple target images
	# 5= color normalization of all images in a folder with multiple target images individually

	#Parameters
	nstains=2    #number of stains
	lamb=0.1     #default value sparsity regularization parameter
	# lamb=0 equivalent to NMF
	
	if op==1:
		filename="Original.png"
		demo_stainsep(filename,nstains,lamb)
	elif op==2:
		level=0
		#output_direc="./"
		output_direc="/home/Drive2/aditya/"
		if not os.path.exists(output_direc):
			os.makedirs(output_direc)
		source_filename="/home/Drive2/aditya/Data/Photos/Training/Benign/b032.png"
		#source_filename="./Test0/20_HE(28016,22316).png"
		#source_filename="./Source/TUPAC-TR-004.svs"
		#source_filename="../../Downloads/01/no1_HE.ndpi"
		#source_filename = "../../Documents/Shubham/Data/Test Data/78/78_HE.ndpi"
		target_filename="/home/aditya/Desktop/Breast_Cancer/Preprocessing/TCGA-E2-A14V-01Z-00-DX1.tif"
		if not os.path.exists(source_filename):
			print "Source file does not exist"
			sys.exit()
		if not os.path.exists(target_filename):
			print "Target file does not exist"
			sys.exit()	
		demo_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level)
	elif op==3:
		level=0


		input_direc="./Test/"
		#input_direc="../../Downloads/01/"
		output_direc="./normalized/"
		if not os.path.exists(output_direc):
			os.makedirs(output_direc)
		file_type="*.png"
		#file_type="*.svs" #all of these file types from input_direc will be normalized
		target_filename="./Target/target.tiff"
		if not os.path.exists(target_filename):
			print "Target file does not exist"
			sys.exit()
		if len(sorted(glob.glob(input_direc+file_type)))==0:
			print "No source files found"
			sys.exit()
		#filename format of normalized images can be changed in demo_colornorm_batch
		filenames=[target_filename]+sorted(glob.glob(input_direc+file_type))


		demo_colornorm_batch(filenames,nstains,lamb,output_direc,level)
	elif op==4:
		level=0

		target_filenames=sorted(glob.glob("./Target/*"))
		print "Target Images:",target_filenames,"\n"
		source_filenames=["./Test0/20_HE(28016,22316).png"]
		print "Source Images:",source_filenames,"\n"
		output_direc="./normalized/"
		if not os.path.exists(output_direc):
			os.makedirs(output_direc)

		for target_filename in target_filenames:
			if not os.path.exists(target_filename):
				print "Target file does not exist"
				continue
			for source_filename in source_filenames:
				if not os.path.exists(source_filename):
					print "Source file does not exist"
					sys.exit()	
				demo_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level)
	elif op==5:
		level=1

		input_direc="./Test/"
		#input_direc="../../Downloads/01/"
		file_type="*"
		#file_type="*.svs" #all of these file types from input_direc will be normalized

		source_filenames=sorted(glob.glob(input_direc+file_type))

		source_filenames=[]
		#source_filenames.append("../../Shubham/Data/Training Data/61/61_HE.ndpi")
		#source_filenames.append("../../Shubham/Data/Training Data/4/4_HE.ndpi")
		source_filenames.append("../../Shubham/Data/Training Data/57/57_HE.ndpi")
		#source_filenames.append("../../Shubham/Data/Test Data/20/20_HE.ndpi")
		#source_filenames.append("../../Shubham/Data/Test Data/37/37_HE.ndpi")
		source_filenames.append("../../Shubham/Data/Test Data/85/85_HE.ndpi")

		if len(source_filenames)==0:
			print "No source files found"
			sys.exit(0)

		for source_filename in source_filenames:
			if not os.path.exists(source_filename):
				print "Source file does not exist:",i
				source_filenames.remove(source_filename)

		print "Source Images:",source_filenames,"\n"

		# target_images_direc="./Target/*"
		# target_filenames=sorted(glob.glob(target_images_direc))

		target_filenames=[]
		#target_filenames.append("../../Shubham/Data/Training Data/61/61_HE.ndpi")
		target_filenames.append("./Target/6_target2.png")
		#target_filenames.append("./Target/6_target2.png")


		print "Target Images:",target_filenames,"\n"

		for target_filename in target_filenames:
			output_direc="./normalized_using_"+os.path.basename(target_filename).replace(".", "_")+"/"
			if not os.path.exists(output_direc):
				os.makedirs(output_direc)
			if not os.path.exists(target_filename):
				print "Target file does not exist"
				continue
			#filename format of normalized images can be changed in demo_colornorm_batch
			filenames=[target_filename]+source_filenames

			demo_colornorm_batch(filenames,nstains,lamb,output_direc,level)

def demo_colornorm_batch(filenames,nstains,lamb,output_direc,img_level):	

	g_1 = tf.Graph()
	with g_1.as_default():
		Wit1=tf.placeholder(tf.float32)
		Wis1=tf.placeholder(tf.float32)
		Img1 = tf.placeholder(tf.float32)
		Hta_Rmax1 = tf.placeholder(tf.float32)
		sav_name = tf.placeholder(tf.string)
		src_i_0 = tf.placeholder_with_default(input=np.array([255.0,255.0,255.0],dtype=np.float32),shape=(3,))
		tar_i_0 = tf.placeholder_with_default(input=np.array([255.0,255.0,255.0],dtype=np.float32),shape=(3,))
		s = tf.shape(Img1)
		Img_vecd = tf.reshape(tf.minimum(Img1,src_i_0),[s[0]*s[1],s[2]])
		V = tf.log(src_i_0+1.0) - tf.log(Img_vecd+1.0)
		Wi_inv = tf.transpose(tf.py_func(np.linalg.pinv, [Wis1], tf.float32))
		Hiv1 = tf.nn.relu(tf.matmul(V,Wi_inv))
		Hso_Rmax = tf.contrib.distributions.percentile(Hiv1,q=99.,axis=[0])
		normfac=tf.divide(Hta_Rmax1,Hso_Rmax)
		Hsonorm=Hiv1*normfac
		source_norm = tf.cast(tar_i_0*tf.exp((-1)*tf.reshape(tf.matmul(Hsonorm,Wit1),(s[0],s[1],s[2]))),tf.uint8)
		enc = tf.image.encode_png(source_norm)
		fwrite = tf.write_file(sav_name,enc)

	g_2 = tf.Graph()
	with g_2.as_default():
		Hso=tf.placeholder(tf.float32)
		Wita=tf.placeholder(tf.float32)
		normfaca=tf.placeholder(tf.float32)
		s1=tf.placeholder(tf.int32)
		tari0 = tf.placeholder_with_default(input=np.array([255.0,255.0,255.0],dtype=np.float32),shape=(3,))
		Hsonorm=Hso*normfaca
		source_norm = tf.cast(tari0*tf.exp((-1)*tf.reshape(tf.matmul(Hsonorm,Wita),tf.stack([s1[0],s1[1],s1[2]]))),dtype=tf.uint8)

	session1=tf.Session(graph=g_1,config=config)
	session2=tf.Session(graph=g_2,config=config)

	file_no=0
	print "To be normalized:",filenames[1:],"using",filenames[0]
	for filename in filenames:

		base_t=os.path.basename(filenames[0]) #target.svs
		fname_t=os.path.splitext(base_t)[0]   #target
		base_s=os.path.basename(filename)     #source.svs
		fname_s=os.path.splitext(base_s)[0]	  #source
		f_form = os.path.splitext(base_s)[1]  #.svs
		s=output_direc+base_s.replace(".", "_")+" (3-channel using "+base_t.replace(".", "_")+").png"
		# s=output_direc+base_s.replace(".", "_")+" (no-norm using "+base_t.replace(".", "_")+").png"
		#s=output_direc+fname_s+"_normalized.png"


		tic=time.time()
		print

		I = openslide.open_slide(filename)
		if img_level>=I.level_count:
			print "Level",img_level,"unavailable for image, proceeding with level 0"
			level=0
		else:
			level=img_level
		xdim,ydim=I.level_dimensions[level]
		ds=I.level_downsamples[level]

		if file_no==0:
			print "Target Stain Separation in progress:",filename,str(xdim)+str("x")+str(ydim)
		else:
			print "Source Stain Separation in progress:",filename,str(xdim)+str("x")+str(ydim)
		print "\t \t \t \t \t \t \t \t \t \t Time: 0"


		#parameters for W estimation
		num_patches=20
		patchsize=1000 #length of side of square 

		Wi,i0 = Wfast(I,nstains,lamb,num_patches,patchsize,level)
		if i0 is None:
			print "No white background detected"
			i0=255

		if Wi is None:
			print "Color Basis Matrix Estimation failed...image normalization skipped"
			continue
		print "W estimated",
		print "\t \t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
		Wi=Wi.astype(np.float32)

		i0_default=np.array([255.,255.,255.],dtype=np.float32)

		if file_no==0:
			print "Target Color Basis Matrix:"
			print Wi
			Wi_target=np.transpose(Wi)
			tar_i0=i0_default
			print "Target Image Background white intensity:",tar_i0
		else:
			print "Source Color Basis Matrix:"
			print Wi
			src_i0=i0_default
			print "Source Image Background white intensity:",src_i0

		_max=1000
		
		print
		if (xdim*ydim)<=(_max*_max):
			print "Small image processing..."
			img=np.asarray(I.read_region((0,0),level,(xdim,ydim)),dtype=np.float32)[:,:,:3]
			
			if file_no==0:
				file_no+=1
				Hiv=session1.run(Hiv1,feed_dict={Wit1: Wi_target, Img1:img, Wis1: Wi, Hta_Rmax1:1.0,sav_name:"tmp",src_i_0:tar_i0})
				Hta_Rmax = np.percentile(Hiv,q=99.,axis=0)
				print "Target H calculated",
				print "\t \t \t \t \t \t \t Total Time:",round(time.time()-tic,3)
				continue

			print "Color Normalization in progress..."
			
			session1.run(fwrite,feed_dict={Wit1: Wi_target, Img1:img, Wis1: Wi, Hta_Rmax1:Hta_Rmax,sav_name:s,src_i_0:src_i0,tar_i_0:tar_i0})

			print "File written to:",s
			print "\t \t \t \t \t \t \t \t \t Total Time:",round(time.time()-tic,3)

		else:
			_maxtf=3000
			x_max=xdim
			y_max=min(max(int(_maxtf*_maxtf/x_max),1),ydim)
			print "Large image processing..."
			if file_no==0:
				Hivt=np.memmap('H_target', dtype='float32', mode='w+', shape=(xdim*ydim,2))
			else:
				Hivs=np.memmap('H_source', dtype='float32', mode='w+', shape=(xdim*ydim,2))
				sourcenorm=np.memmap('wsi', dtype='uint8', mode='w+', shape=(ydim,xdim,3))
			x_tl = range(0,xdim,x_max)
			y_tl = range(0,ydim,y_max)
			print "WSI divided into",str(len(x_tl))+"x"+str(len(y_tl))
			count=0
			print "Patch-wise H calculation in progress..."
			ind=0
			perc=[]
			for x in x_tl:
				for y in y_tl:
					count+=1
					xx=min(x_max,xdim-x)
					yy=min(y_max,ydim-y)
					print "Processing:",count,"		patch size",str(xx)+"x"+str(yy),
					print "\t \t Time since processing started:",round(time.time()-tic,3)
					img=np.asarray(I.read_region((int(ds*x),int(ds*y)),level,(xx,yy)),dtype=np.float32)[:,:,:3]		

					if file_no==0:
						Hiv = session1.run(Hiv1,feed_dict={Wit1: Wi_target, Img1:img, Wis1: Wi, Hta_Rmax1:1.0,sav_name:"tmp",src_i_0:tar_i0})
						Hivt[ind:ind+len(Hiv),:]=Hiv
						_Hta_Rmax = np.percentile(Hiv,q=99.,axis=0)
						perc.append([_Hta_Rmax[0],_Hta_Rmax[1]])
						ind+=len(Hiv)
						continue
					else:
						Hiv = session1.run(Hiv1,feed_dict={Wit1: Wi_target, Img1:img, Wis1: Wi, Hta_Rmax1:1.0,sav_name:"tmp",src_i_0:src_i0,tar_i_0:tar_i0})
						_Hso_Rmax = np.percentile(Hiv,q=99.,axis=0)
						Hivs[ind:ind+len(Hiv),:]=Hiv
						perc.append([_Hso_Rmax[0],_Hso_Rmax[1]])
						ind+=len(Hiv)

			if file_no==0:
				print "Target H calculated",
				Hta_Rmax = np.percentile(np.array(perc),50,axis=0)
				file_no+=1
				del Hivt
				print "\t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
				ind=0
				continue

			print "Source H calculated",
			print "\t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
			Hso_Rmax = np.percentile(np.array(perc),50,axis=0)
			print "H Percentile calculated", 
			print "\t \t \t \t Time since processing started:",round(time.time()-tic,3)

			_normfac=np.divide(Hta_Rmax,Hso_Rmax).astype(np.float32)

			print "Color Normalization in progress..."
			count=0
			ind=0
			np_max=1000

			x_max=xdim
			y_max=min(max(int(np_max*np_max/x_max),1),ydim)
			x_tl = range(0,xdim,x_max)
			y_tl = range(0,ydim,y_max)
			print "Patch-wise color normalization in progress..."
			total=len(x_tl)*len(y_tl)
			milestone=5
			for x in x_tl:
				for y in y_tl:
					count+=1
					xx=min(x_max,xdim-x)
					yy=min(y_max,ydim-y)
					pix=xx*yy
					sh=np.array([yy,xx,3])
					
					#Back projection into spatial intensity space (Inverse Beer-Lambert space)
					
					sourcenorm[y:y+yy,x:x+xx,:3]=session2.run(source_norm,feed_dict={Hso: np.array(Hivs[ind:ind+pix,:]),Wita:Wi_target,normfaca:_normfac,s1:sh,tari0:tar_i0})

					ind+=pix
					percent=int(count*100/total)
					if percent==milestone:
						if milestone==100:
							print "Color Normalization complete!",
							print "\t \t \t \t Time since processing started:",round(time.time()-tic,3)
						else:
							print str(percent)+" percent complete...",
							print "\t \t \t \t \t Time since processing started:",round(time.time()-tic,3)
							milestone+=5

			print "Saving normalized image..."
			cv2.imwrite(s,cv2.cvtColor(sourcenorm, cv2.COLOR_RGB2BGR))
			del sourcenorm
			print "File written to:",s
			print "\t \t \t \t \t \t \t \t \t Total Time:",round(time.time()-tic,3)

		file_no+=1
		if os.path.exists("H_target"):
			os.remove("H_target")
		if os.path.exists("H_source"):
			os.remove("H_source")
		if os.path.exists("wsi"):
			os.remove("wsi")

	session1.close()
	session2.close()

def demo_colornorm(source_filename,target_filename,nstains,lamb,output_direc,level):	
	filenames=[target_filename,source_filename]
	demo_colornorm_batch(filenames,nstains,lamb,output_direc,level)

def patch_Valid(patch,threshold):
	r_th=220
	g_th=220
	b_th=220
	tempr = patch[:,:,0]>r_th
	tempg = patch[:,:,1]>g_th
	tempb = patch[:,:,2]>b_th
	temp = tempr*tempg*tempb
	r,c = np.shape((temp)) 
	prob= float(np.sum(temp))/float((r*c))
	#print prob
	if prob>threshold:
		return False
	else:
		return True  

def Wfast(img,nstains,lamb,num_patches,patchsize,level):
	
	param=definePar(nstains,lamb)
	_max=2000
	max_size=_max*_max
	xdim,ydim=img.level_dimensions[0]
	patchsize=int(min(patchsize,xdim/3,ydim/3))
	patchsize_original=patchsize
	nstains=param['K']
	valid_inp=[]
	
	white_pixels=[]
	num_white=min(100000,xdim*ydim)
	white_cutoff=220
	I_percentile=80

	if ydim*xdim>max_size:
		print "Finding patches for W estimation:"
		for j in range(20):
			#print "Patch Sampling Attempt:",i+1
			initBias=int(math.ceil(patchsize/2)+1) 
			xx=np.array(range(initBias,xdim-initBias,patchsize))
			yy=np.array(range(initBias,ydim-initBias,patchsize))
			xx_yy=np.transpose([np.tile(xx, len(yy)), np.repeat(yy, len(xx))])
			np.random.shuffle(xx_yy)

			threshold=0.1 #maximum percentage of white pixels in patch
			for i in range(len(xx_yy)):
				patch=np.asarray(img.read_region((xx_yy[i][0],xx_yy[i][1]),level,(patchsize,patchsize)))
				patch=patch[:,:,:3]
				if len(white_pixels)<num_white:
					white_pixels.extend(patch[np.sum((patch>white_cutoff),axis=2)==3])

				if patch_Valid(patch,threshold):
					valid_inp.append(patch)
					if len(valid_inp)==num_patches:
						break

			if len(valid_inp)==num_patches:
				white_pixels=np.array(white_pixels[:num_white])
				break																																																																																																																																	
			patchsize=int(patchsize*0.95)
		valid_inp=np.array(valid_inp)
		print "Number of patches sampled for W estimation:", len(valid_inp)
	else:
		patch=np.asarray(img.read_region((0,0),level,(xdim,ydim)))
		valid_inp=[]
		valid_inp.append(patch)
		white_pixels= patch[np.sum((patch>white_cutoff),axis=2)==3]
		print "Image small enough...W estimation done using whole image"

	# print white_pixels
	if len(white_pixels)>0:
		# print len(white_pixels)
		i0 = np.percentile(white_pixels,I_percentile,axis=0)[:3]
	else:
		i0 = None


	if len(valid_inp)>0:
		out = suppress_stdout()
		pool = Pool(initializer=initializer)
		try:
		    WS_tmp = pool.map(partial(getstainMat,param=param),valid_inp)
		except KeyboardInterrupt:
			pool.terminate()
			pool.join()
		pool.terminate()
		pool.join()
		suppress_stdout(out)

		#Discard WS where all are zeros
		WS=[]
		for i in WS_tmp:
			if np.sum(i)!=0:
				WS.append(i)
		WS=np.array(WS)

		if len(WS)<0.5*len(WS_tmp):
			print "Less than half the sampled patches gave valid color basis...trying again..."
			return Wfast(img,nstains,lamb,min(100,num_patches*1.5),int(patchsize_original),level)
		elif len(WS_tmp)!=1:
			print "Median color basis of",len(WS),"patches used as actual color basis"

		if WS.shape[0]==1:
			Wsource=WS[0,:3,:]
		else:
			Wsource=np.zeros((3,nstains))
			for k in range(nstains):
			    Wsource[:,k]=[np.median(WS[:,0,k]),np.median(WS[:,1,k]),np.median(WS[:,2,k])]
		
		Wsource=normalize_W(Wsource,nstains)
		tmp=np.transpose(Wsource)
		Wsource = np.transpose(tmp[tmp[:,2].argsort(),]) #Sorting of color bases
		
		if Wsource.sum()==0:
			if patchsize*0.95<100:
				print "No suitable patches found for learning W. Please relax constraints"
				return None			#to prevent infinite recursion
			else:
				print "W estimation failed, matrix of all zeros found. Trying again..."				
				return Wfast(img,nstains,lamb,min(100,num_patches*1.5),int(patchsize_original*0.95),level)
		else:
			return Wsource,i0
	else:
		print "No suitable patches found for learning W. Please relax constraints"
		return None,None

def BLtrans(Ivecd):
	Ivecd=vectorise(Ivecd)
	V=np.log(256)- np.log(Ivecd+1.0)
	w_threshold=220
	c = (Ivecd[:,0]<w_threshold) * (Ivecd[:,1]<w_threshold) * (Ivecd[:,2]<w_threshold)
	Ivecd=Ivecd[c]
	VforW=np.log(256)- np.log(Ivecd+1.0) #V=WH, +1 is to avoid divide by zero
	#shape of V = no. of pixels x 3 
	return V,VforW

def BLtrans2(Ivecd):
	Ivecd=vectorise(Ivecd)
	V=np.log(256)- np.log(Ivecd+1.0) #V=WH, +1 is to avoid divide by zero
	#shape of V = no. of pixels x 3 
	return V

def getstainMat(I,param):
	#Source Input
	#I : Patch for W estimation
	V,VforW=BLtrans(I)   #Beer-Lambert law
	#step 2: Sparse NMF factorization (Learning W; V=WH)
	out = suppress_stdout()
	Ws = spams.trainDL(np.asfortranarray(np.transpose(VforW)),**param)
	suppress_stdout(out)
	#Label the columns to be Hematoxylin and eosin
	tmp=np.transpose(Ws)
	Ws = np.transpose(tmp[tmp[:,2].argsort(),])
	return Ws

def definePar(nstains,lamb,batch=None):

	param={}	
	#param['mode']=2               #solves for =min_{D in C} (1/n) sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2 + ... 
								   #lambda||alpha_i||_1 + lambda_2||alpha_i||_2^2
	param['lambda1']=lamb
	#param['lambda2']=0.05
	param['posAlpha']=True         #positive stains 
	param['posD']=True             #positive staining matrix
	param['modeD']=0               #{W in Real^{m x n}  s.t.  for all j,  ||d_j||_2^2 <= 1 }
	param['whiten']=False          #Do not whiten the data                      
	param['K']=nstains             #No. of stain = 2
	param['numThreads']=-1         #number of threads
	param['iter']=40               #20-50 is OK
	param['clean']=True
	if batch is not None:
		param['batchsize']=batch   #Give here input image no of pixels for traditional dictionary learning
	return param

def normalize_W(W,k):
	W1 = preprocessing.normalize(W, axis=0, norm='l2')
	return W1

def vectorise(I):
	s=I.shape
	if len(s)==2: #case for 2D array
		third_dim=1
	else:
		third_dim=s[2] 
	return np.reshape(I, (s[0]*s[1],third_dim))

def suppress_stdout(out=None):
	if out is None:
		devnull = open('/dev/null', 'w')
		oldstdout_fno = os.dup(sys.stdout.fileno())
		os.dup2(devnull.fileno(), 1)
		return oldstdout_fno
	else:
		os.dup2(out, 1)

def demo_stainsep(filename,nstains,lamb):
	#Demo of Fast Sparse Stain Separation (SSS)
	#Read input image
	
	level=0

	I = openslide.open_slide(filename)
	xdim,ydim=I.level_dimensions[level]
	img=np.asarray(I.read_region((0,0),level,(xdim,ydim)))[:,:,:3]

	print "Fast stain separation is running...."
	t = time.time()
	Wi,Hi,Hiv,stains=Faststainsep(I,img,nstains,lamb,level)
	elapsed = time.time() - t


	print "\t \t \t \t \t \t Time taken:",elapsed

	print "Color Basis Matrix:\n",Wi

	f, axarr = plt.subplots(1, nstains+1, sharey=True)
	im0 = axarr[0].imshow(img,aspect='auto')
	axarr[0].set_title('Original Image')
	for i in range(1,nstains+1):
		axarr[i].imshow(stains[i-1],aspect='auto')
		axarr[i].set_title('Stain '+str(i))
	plt.show()

def Faststainsep(I_obj,I,nstains,lamb,level):
	s=I.shape
	ndimsI = len(s)
	if ndimsI!=3:
		print "Input Image I should be 3-dimensional!"
		sys.exit(0)
	rows = s[0]
	cols = s[1]

	num_patches=20
	patchsize=100

	#Beer-Lambert tranformation
	V,VforW=BLtrans(I)    #V=WH see in paper
	#Estimate stain color bases + acceleration
	Wi,_=Wfast(I_obj,nstains,lamb,num_patches,patchsize,level)      
	Hiv=np.transpose(np.dot(linalg.pinv(Wi),np.transpose(V)))  #Pseudo-inverse
	Hiv[Hiv<0]=0

	Hi=np.reshape(Hiv,(rows,cols,nstains))
	#calculate the color image for each stain
	sepstains = []
	for i in range(nstains):
		vdAS =  np.reshape(Hiv[:,i],(rows*cols,1))*np.reshape(Wi[:,i],(1,3))
		sepstains.append(np.uint8(255*np.reshape(np.exp(-vdAS), (rows, cols, 3))))
	return Wi,Hi,Hiv,sepstains

def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

main()
