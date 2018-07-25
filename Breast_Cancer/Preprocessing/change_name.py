import cv2
import glob

dirlist = ['/home/Drive2/aditya/Data/Photos/Training/Benign/*tif',
'/home/Drive2/aditya/Data/Photos/Training/InSitu/*tif',
'/home/Drive2/aditya/Data/Photos/Training/Invasive/*tif',
'/home/Drive2/aditya/Data/Photos/Training/Normal/*tif',
'/home/Drive2/aditya/Data/Photos/Validation/Benign/*tif',
'/home/Drive2/aditya/Data/Photos/Validation/InSitu/*tif',
'/home/Drive2/aditya/Data/Photos/Validation/Invasive/*tif',
'/home/Drive2/aditya/Data/Photos/Validation/Normal/*tif']


for m in range(len(dirlist)):
    ct=1
    for filename in sorted(glob.glob(dirlist[m])):
        image_name = filename[int(len(filename)-filename[::-1].find("/")):filename.find(".")]
        img = cv2.imread(filename)
        cv2.imwrite(dirlist[m][0:-4]+image_name+".png",img)
        print(m,ct,image_name)
        ct+=1
