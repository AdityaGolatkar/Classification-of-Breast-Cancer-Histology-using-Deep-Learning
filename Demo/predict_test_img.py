from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from test_preprocess import test_pre_process
import pdb
import shutil
import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#######################################################
#Uncomment these lines if you want to limit the gpu use
#######################################################
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.33
#config.gpu_options.visible_device_list = "1"
#set_session(tf.Session(config=config))


def predict_test_image(test_image_path):

    ########################################
    #Initialization
    ########################################
    img_width, img_height = 299, 299
    cwd = os.getcwd()
    test_data_dir = cwd+'/Test_img_folder/'
    model_load_weights = [cwd+'/final_weights_1.h5',cwd+'/final_weights_2.h5']
    ########################################

    ##############################
    #Nuclei based Patch Extraction
    ##############################
    no_of_patches, image_name = test_pre_process(test_image_path)

    #########################################
    #Defining our modified Inception-v3 model
    #########################################
    input_shape = (img_width, img_height, 3)

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu', name='fool_con1')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(4, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

  
    ########################
    #Patch based preditcions
    ########################

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        seed=0)

    class_vec = np.zeros((no_of_patches,4))
    for w in range(len(model_load_weights)):
        model.load_weights(model_load_weights[w])
        result = model.predict_generator(
            test_generator,
            steps=len(test_generator.filenames)//1,
            verbose=0)
        
        class_vec+=result
    class_vec = 1.0*class_vec/len(model_load_weights)
    
    #pdb.set_trace()
    class_vector = np.zeros((4,1))
    for i in range(class_vec.shape[0]):
        cancer_class = np.argmax(class_vec[i,:])
        class_vector[cancer_class]+=1

    image_class_loc = np.argmax(class_vector)
    
    if class_vector[3] == class_vector[image_class_loc]:
        image_class = 3
    if class_vector[0] == class_vector[image_class_loc]:
        image_class = 0
    if class_vector[1] == class_vector[image_class_loc]:
        image_class = 1
    if class_vector[2] == class_vector[image_class_loc]:
        image_class = 2

    if image_class == 0:
        cancer_grade = 'Benign tumor'
    elif image_class == 1:
        cancer_grade = 'In-Situ Carcinoma'
    elif image_class == 2:
        cancer_grade = 'Invasive Carcinoma'
    elif image_class == 3:
        cancer_grade = 'Normal tissue'
    print("Here is our prediction :")
    print("The patient "+image_name+" belongs to "+cancer_grade+" class")

    shutil.rmtree("Test_img_folder/")


parser = argparse.ArgumentParser(description='Image Path')
parser.add_argument('text', action='store', type=str, help='Image Path')
args = parser.parse_args()
predict_test_image(args.text)
