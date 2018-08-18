from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import pdb
import pickle
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))


####################################################################
no_samples = 25

img_width, img_height = 299, 299
filenames_path = '/home/Drive2/aditya/filenames.txt'
result_path = '/home/Drive2/aditya/result.npy'
train_data_dir = '/home/Drive2/aditya/Aug_Data/Training/'
validation_data_dir = '/home/Drive2/aditya/Aug_Data/Validation/'

initial_weights = "/home/Drive2/aditya/Weights/init_weight.h5"
last_epoch_weights = "/home/Drive2/aditya/Weights/last_weights.h5"
best_epoch_weights = "/home/Drive2/aditya/Weights/fold2_weights.h5"
model_load_weights = best_epoch_weights
#####################################################################


input_shape = (img_width, img_height, 3)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu', name='fool_con1')(x)
# and a logistic layer -- let's say we have 200 classes
#x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights(model_load_weights)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=1,
    shuffle=False,
    seed=0)

result = model.predict_generator(
        test_generator,
        steps=len(test_generator.filenames)//1,
        verbose=1)

np.save(result_path,result)
#np.save(filenames_path,test_generator.filenames)
with open(filenames_path, 'wb') as f:
    pickle.dump(test_generator.filenames, f)

"""
no_samples = 25
no_subsamples = 129

benign = -np.ones((no_samples,no_subsamples))
insitu = -np.ones((no_samples,no_subsamples))
invasive = -np.ones((no_samples,no_subsamples))
normal = -np.ones((no_samples,no_subsamples))

for i in range(len(result)):    
    result_vector = result[i,:]
    full_image_classes = np.zeros((4,1))
    cancer_class = np.argmax(result_vector)
    
    file_name = test_generator.filenames[i]
    full_image_name = int(file_name[file_name.find("/")+1:file_name.find("_")])
    patch_name = int(file_name[file_name.find("_")+1:file_name.find(".")])
    
    if i < no_samples*no_subsamples:
        benign[full_image_name-1,patch_name-1] = cancer_class        
    elif i >= no_samples*no_subsamples and i < 2*no_samples*no_subsamples:
        insitu[full_image_name-1,patch_name-1] = cancer_class 
    elif i >= 2*no_samples*no_subsamples and i < 3*no_samples*no_subsamples:
        invasive[full_image_name-1,patch_name-1] = cancer_class 
    elif i >= 3*no_samples*no_subsamples and i < 4*no_samples*no_subsamples:
        normal[full_image_name-1,patch_name-1] = cancer_class

pdb.set_trace()

np.save("/home/Drive2/aditya/Benign.npy",benign)
np.save("/home/Drive2/aditya/InSitu.npy",insitu)
np.save("/home/Drive2/aditya/Invasive.npy",invasive)
np.save("/home/Drive2/aditya/Normal.npy",normal)
"""
