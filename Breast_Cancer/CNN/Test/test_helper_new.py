import pdb
import numpy as np
import pickle
#################################################################################
no_samples = 25

result = np.load('/home/Drive2/aditya/result.npy')
with open('/home/Drive2/aditya/filenames.txt', 'rb') as f:
    filenames = pickle.load(f)
benign_subsamples  = np.load("/home/Drive2/aditya/benign_patches_info.npy")
normal_subsamples  = np.load("/home/Drive2/aditya/normal_patches_info.npy")
insitu_subsamples  = np.load("/home/Drive2/aditya/insitu_patches_info.npy")
invasive_subsamples  = np.load("/home/Drive2/aditya/invasive_patches_info.npy")

benign_tot_ss = np.sum(benign_subsamples)
insitu_tot_ss = np.sum(insitu_subsamples)
invasive_tot_ss = np.sum(invasive_subsamples)
normal_tot_ss = np.sum(normal_subsamples)
################################################################################

benign = []
insitu = []
invasive = []
normal = []

for i in range(no_samples):
    if benign_subsamples[i] == 0:
        benign.append(-np.ones((int(benign_subsamples[i]+1),1)))
    else:
        benign.append(-np.ones((int(benign_subsamples[i]),1)))
    if insitu_subsamples[i] == 0:
        insitu.append(-np.ones((int(insitu_subsamples[i]+1),1)))
    else:
        insitu.append(-np.ones((int(insitu_subsamples[i]),1))) 
    if invasive_subsamples[i] == 0:
        invasive.append(-np.ones((int(invasive_subsamples[i]+1),1)))
    else:
        invasive.append(-np.ones((int(invasive_subsamples[i]),1)))
    if normal_subsamples[i] == 0:
        normal.append(-np.ones((int(normal_subsamples[i]+1),1)))
    else:
        normal.append(-np.ones((int(normal_subsamples[i]),1)))

for i in range(len(result)):
    result_vector = result[i,:]
    cancer_class = np.argmax(result_vector)
    
    file_name = filenames[i]
    full_image_name = int(file_name[file_name.find("/")+1:file_name.find("_")])
    patch_name = int(file_name[file_name.find("_")+1:file_name.find(".")])
    
    if i < benign_tot_ss:
        benign[full_image_name][patch_name] = cancer_class

    elif i >= benign_tot_ss and i < benign_tot_ss + insitu_tot_ss:
        insitu[full_image_name][patch_name] = cancer_class

    elif i >= benign_tot_ss + insitu_tot_ss and i < benign_tot_ss + insitu_tot_ss + invasive_tot_ss:
        invasive[full_image_name][patch_name] = cancer_class

    elif i >= benign_tot_ss + insitu_tot_ss + invasive_tot_ss and i < benign_tot_ss + insitu_tot_ss + invasive_tot_ss + normal_tot_ss:
        normal[full_image_name][patch_name] = cancer_class

pdb.set_trace()


with open("/home/Drive2/aditya/Benign.txt", 'wb') as f:
    pickle.dump(benign, f)
with open("/home/Drive2/aditya/InSitu.txt", 'wb') as f:
    pickle.dump(insitu, f)
with open("/home/Drive2/aditya/Invasive.txt", 'wb') as f:
    pickle.dump(invasive, f)
with open("/home/Drive2/aditya/Normal.txt", 'wb') as f:
    pickle.dump(normal, f)
#np.save("/home/Drive2/aditya/Benign.txt",benign)
#np.save("/home/Drive2/aditya/InSitu.txt",insitu)
#np.save("/home/Drive2/aditya/Invasive.txt",invasive)
#np.save("/home/Drive2/aditya/Normal.txt",normal)
