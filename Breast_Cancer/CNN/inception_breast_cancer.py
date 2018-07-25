from __future__ import division
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import pdb
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))

####################################################################
#Parameters
####################################################################
img_width, img_height = 299, 299
train_data_dir = '/home/Drive2/aditya/Aug_Data/Training/'
validation_data_dir = '/home/Drive2/aditya/Aug_Data/Validation/'
best_epoch_weights = "/home/Drive2/aditya/Weights/fold2_weights.h5"
last_epoch_weights = "/home/Drive2/aditya/Weights/semifinal_fold2_weights.h5"
initial_weights = "/home/Drive2/aditya/Weights/fold2_weights.h5"
batch_size = 32
epochs1 = 2
epochs2 = 2
####################################################################

benign_train_no = len(os.listdir(train_data_dir+"Benign/"))
insitu_train_no = len(os.listdir(train_data_dir+"InSitu/"))
invasive_train_no = len(os.listdir(train_data_dir+"Invasive/"))
normal_train_no = len(os.listdir(train_data_dir+"Normal/"))
total_train_no = benign_train_no + insitu_train_no + invasive_train_no + normal_train_no

nb_train_samples = total_train_no

benign_validation_no = len(os.listdir(validation_data_dir+"Benign/"))
insitu_validation_no = len(os.listdir(validation_data_dir+"InSitu/"))
invasive_validation_no = len(os.listdir(validation_data_dir+"Invasive/"))
normal_validation_no = len(os.listdir(validation_data_dir+"Normal/"))
total_val_no = benign_validation_no + insitu_validation_no + invasive_validation_no + normal_validation_no

nb_validation_samples = total_val_no

class_weight = {0:max(1.0,1.0*total_train_no/benign_train_no),1:max(1.0,1.0*total_train_no/insitu_train_no),2:max(1.0,1.0*total_train_no/invasive_train_no),3:max(1.0,1.0*total_train_no/normal_train_no)}

input_shape = (img_width, img_height, 3)

checkpointer = ModelCheckpoint(filepath= best_epoch_weights, monitor='val_acc',verbose=1, save_best_only=True,save_weights_only=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    seed=0)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    seed=0)


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

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

#model.summary()

model.load_weights(initial_weights)

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs1,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[checkpointer],
    class_weight=class_weight)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs2,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[checkpointer],
    class_weight=class_weight)

model.save_weights(last_epoch_weights)
"""
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    shuffle=False,
    seed=0)

result = model.predict_generator(
        test_generator,
        steps=nb_validation_samples//batch_size)
"""

#pdb.set_trace()
