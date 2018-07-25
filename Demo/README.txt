
This folder contains two files:
1)predict_test_img : This will predict the class of a test image using our proposed method.
2)test_preprocess : This will perform the necessary image preprocessing.


#######################################################
To run these files the following packages are REQUIRED:
#######################################################
python
keras
pdb
shutil
argparse
os
pickle
numpy 
tensorflow
openCV2


######
USAGE:
######

python predict_test_img.py "path to the image"

eg. python predict_test_img.py "/home/Desktop/test_img.tif"

It is necessary to specify the complete path to the image. Also the path should be enclosed within double quotes.


###############################
The output will be of the form:
##############################

Here is our prediction :
The patient n027 belongs to Normal tissue class




