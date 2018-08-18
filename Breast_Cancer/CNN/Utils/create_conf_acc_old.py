import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import pdb
import shutil

######################################################################
#Parameters
######################################################################
benign_ss_path  = "/home/Drive2/aditya/benign_patches_info.npy"
normal_ss_path  = "/home/Drive2/aditya/normal_patches_info.npy"
insitu_ss_path  = "/home/Drive2/aditya/insitu_patches_info.npy"
invasive_ss_path  = "/home/Drive2/aditya/invasive_patches_info.npy"

benign_ss = np.load(benign_ss_path)
insitu_ss = np.load(insitu_ss_path)
invasive_ss = np.load(invasive_ss_path)
normal_ss = np.load(normal_ss_path)

with open('/home/Drive2/aditya/Benign.txt', 'rb') as f:
    benign = pickle.load(f)
with open('/home/Drive2/aditya/InSitu.txt', 'rb') as f:
    insitu = pickle.load(f)
with open('/home/Drive2/aditya/Invasive.txt', 'rb') as f:
    invasive = pickle.load(f)
with open('/home/Drive2/aditya/Normal.txt', 'rb') as f:
    normal = pickle.load(f)

#benign = np.load('/home/Drive2/aditya/Benign.npy')
#insitu = np.load('/home/Drive2/aditya/InSitu.npy')
#invasive = np.load('/home/Drive2/aditya/Invasive.npy')
#normal = np.load('/home/Drive2/aditya/Normal.npy')
no_samples = 25
######################################################################
#pdb.set_trace()

benign_hist = np.zeros((no_samples,4))
insitu_hist = np.zeros((no_samples,4))
invasive_hist = np.zeros((no_samples,4))
normal_hist = np.zeros((no_samples,4))

confusion_matrix = np.zeros((4,4))

benign_conf = np.zeros((4,1))
insitu_conf = np.zeros((4,1))
invasive_conf = np.zeros((4,1))
normal_conf = np.zeros((4,1))

# 0: Benign, 1:InSitu, 2:Invasive, 3:Normal
for i in range(no_samples):
    
    benign_hist[i,0] = (benign[i][:] == 0).sum()
    benign_hist[i,1] = (benign[i][:] == 1).sum()
    benign_hist[i,2] = (benign[i][:] == 2).sum()
    benign_hist[i,3] = (benign[i][:] == 3).sum()
    benign_hist[i,:] = benign_hist[i,:]/len(benign[i][:])
    #print(np.sum(benign_hist[i,:]))

    benign_class_temp = np.argmax(benign_hist[i,:])

    if benign_hist[i,3] == benign_hist[i,benign_class_temp]:
        benign_class = 3
    if benign_hist[i,0] == benign_hist[i,benign_class_temp]:
        benign_class = 0
    if benign_hist[i,1] == benign_hist[i,benign_class_temp]:
        benign_class = 1
    if benign_hist[i,2] == benign_hist[i,benign_class_temp]:
        benign_class = 2

    benign_conf[benign_class]+=1

    insitu_hist[i,0] = (insitu[i][:] == 0).sum()
    insitu_hist[i,1] = (insitu[i][:] == 1).sum()
    insitu_hist[i,2] = (insitu[i][:] == 2).sum()
    insitu_hist[i,3] = (insitu[i][:] == 3).sum()
    insitu_hist[i,:] = insitu_hist[i,:]/len(insitu[i][:])
    #print(np.sum(insitu_hist[i,:]))

    insitu_class_temp = np.argmax(insitu_hist[i,:])

    if insitu_hist[i,3] == insitu_hist[i,insitu_class_temp]:
        insitu_class = 3
    if insitu_hist[i,0] == insitu_hist[i,insitu_class_temp]:
        insitu_class = 0
    if insitu_hist[i,1] == insitu_hist[i,insitu_class_temp]:
        insitu_class = 1
    if insitu_hist[i,2] == insitu_hist[i,insitu_class_temp]:
        insitu_class = 2

    insitu_conf[insitu_class]+=1

    if insitu_class == 0 or insitu_class == 3 or insitu_class == 2:
        for j in range(insitu_ss[i]):
            shutil.copyfile("/home/Drive2/aditya/Aug_Data/Validation/InSitu/"+str(i)+"_"+str(j)+".png","/home/Drive2/aditya/Wrong_Class/InSitu/"+str(i)+"_"+str(j)+"_TC_"+str(insitu_class)+"_AC_1_PC_"+str(int(insitu[i][j]))+".png")
        print(" InSitu " + str(i) + " incorrectly classified as " + str(insitu_class))
    

    invasive_hist[i,0] = (invasive[i][:] == 0).sum()
    invasive_hist[i,1] = (invasive[i][:] == 1).sum()
    invasive_hist[i,2] = (invasive[i][:] == 2).sum()
    invasive_hist[i,3] = (invasive[i][:] == 3).sum()
    invasive_hist[i,:] = invasive_hist[i,:]/len(invasive[i][:])
    #print(np.sum(invasive_hist[i,:]))

    #invasive_class = np.argmax(invasive_hist[i,:])
    invasive_class_temp = np.argmax(invasive_hist[i,:])

    if invasive_hist[i,3] == invasive_hist[i,invasive_class_temp]:
        invasive_class = 3
    if invasive_hist[i,0] == invasive_hist[i,invasive_class_temp]:
        invasive_class = 0
    if invasive_hist[i,1] == invasive_hist[i,invasive_class_temp]:
        invasive_class = 1
    if invasive_hist[i,2] == invasive_hist[i,invasive_class_temp]:
        invasive_class = 2

    invasive_conf[invasive_class]+=1

    if invasive_class == 0 or invasive_class == 3 or invasive_class == 1:
        for j in range(invasive_ss[i]):
            shutil.copyfile("/home/Drive2/aditya/Aug_Data/Validation/Invasive/"+str(i)+"_"+str(j)+".png","/home/Drive2/aditya/Wrong_Class/Invasive/"+str(i)+"_"+str(j)+"_TC_"+str(invasive_class)+"_AC_2_PC_"+str(int(invasive[i][j]))+".png")       

        print(" Invasive " + str(i) + " incorrectly classified as " +str(invasive_class))
        
    normal_hist[i,0] = (normal[i][:] == 0).sum()
    normal_hist[i,1] = (normal[i][:] == 1).sum()
    normal_hist[i,2] = (normal[i][:] == 2).sum()
    normal_hist[i,3] = (normal[i][:] == 3).sum()
    normal_hist[i,:] = normal_hist[i,:]/len(normal[i][:])
    #print(np.sum(normal_hist[i,:]))

    #normal_class = np.argmax(normal_hist[i,:])
    normal_class_temp = np.argmax(normal_hist[i,:])

    if normal_hist[i,3] == normal_hist[i,normal_class_temp]:
        normal_class = 3
    if normal_hist[i,0] == normal_hist[i,normal_class_temp]:
        normal_class = 0
    if normal_hist[i,1] == normal_hist[i,normal_class_temp]:
        normal_class = 1
    if normal_hist[i,2] == normal_hist[i,normal_class_temp]:
        normal_class = 2

    normal_conf[normal_class]+=1

    if normal_class == 0 or normal_class == 2 or normal_class == 1:
        for j in range(normal_ss[i]):
            shutil.copyfile("/home/Drive2/aditya/Aug_Data/Validation/Normal/"+str(i)+"_"+str(j)+".png","/home/Drive2/aditya/Wrong_Class/Normal/"+str(i)+"_"+str(j)+"_TC_"+str(normal_class)+"_AC_3_PC_"+str(int(normal[i][j]))+".png")
        print(" Normal " + str(i) + " incorrectly classified as " +str(normal_class))


print("")
print("Confusion Matrix: 1 : Benign, 2 : InSitu, 3 : Invasive, 4 : Normal")
print("")
#pdb.set_trace()
print("Beni | InSi | Inv | Norm")
confusion_matrix[:,0] = benign_conf[:,0]
confusion_matrix[:,1] = insitu_conf[:,0]
confusion_matrix[:,2] = invasive_conf[:,0]
confusion_matrix[:,3] = normal_conf[:,0]

print(confusion_matrix)
print("")

print("Benign Acc: ", benign_conf[0]/no_samples)
print("Insitu Acc: ", insitu_conf[1]/no_samples)
print("Invasive Acc: ", invasive_conf[2]/no_samples)
print("Normal Acc: ", normal_conf[3]/no_samples)
