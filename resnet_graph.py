#import matplotlib.image as mpimg
import numpy as np      # 1.10.1
import pandas as pd
import glob
import random
import time
from sklearn.cross_validation import train_test_split

from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Graph
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.optimizers import SGD

from helper_functions import mean_f1_score, resnet_image_processing
from helper_functions import show_image_labels

from joblib import Parallel, delayed
import tables


#%% Configuration 
number_of_epochs = 15
n_images = 2000
imsize   = 224      # Square images
save_model = False
show_plots = False
model_name = 'mar_7_0005'
weight_decay = 0.01

#%% Read in the images
# TODO: Run on some Image classification datasets: CIFAR10, MNIST...
#        
# Boids... http://www.vision.caltech.edu/visipedia/CUB-200.html


#%% Make a tensor
tensor = train_images
'''
Reshape to fit Theanos format 
dim_ordering='th'
(samples, channels, rows, columns)
vs
dim_ordering='tf'
(samples, rows, cols, channels)

'''
tensor = tensor.reshape(n_images,3,imsize,imsize)
tensor = tensor.astype('float32')


#%% Final processing and setup

im_mean = tensor.mean()
tensor -= im_mean       # Subtract the mean -- should be per image instead??

train_ind, test_ind, _, _ = train_test_split(range(n_images),
                                             range(n_images),
                                             test_size=0.1, 
                                             random_state=4)

print 'Mean for all images: {}'.format(im_mean)

#%% ResNet-like 
"""18 layer ResNet with skip connections 

    http://arxiv.org/abs/1512.03385
"""
graph = Graph()
graph.add_input(name='input', input_shape=(3,imsize,imsize))   
graph.add_node(Convolution2D(nb_filter=64, nb_row=7, nb_col=7,
                             input_shape=(3,imsize,imsize),
                             border_mode='same',
                             subsample=(2,2),
                             dim_ordering='th',
                             W_regularizer=l2(weight_decay)), 
                             name='conv1', input='input')
graph.add_node(BatchNormalization(), name='bn1', input='conv1')
graph.add_node(Activation('relu'), name='relu1', input='bn1')
graph.add_node(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                            border_mode='same'), 
               name='pool1', input='relu1')
# Output shape = (56,56)x64
               
graph.add_node(Convolution2D(64, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv2_1', input='pool1')
graph.add_node(BatchNormalization(), name='bn2_1', input='conv2_1')
graph.add_node(Activation('relu'), name='relu2_1', input='bn2_1')
graph.add_node(Convolution2D(64, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv2_2', input='relu2_1')
graph.add_node(BatchNormalization(), name='bn2_2', input='conv2_2')
graph.add_node(Activation('relu'), name='relu2_2', inputs=['pool1','bn2_2'], 
               merge_mode='sum')
# Output shape = (64,56,56)
               
graph.add_node(Convolution2D(64, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv2_3', input='relu2_2')
graph.add_node(BatchNormalization(), name='bn2_3', input='conv2_3')
graph.add_node(Activation('relu'), name='relu2_3', input='bn2_3')
graph.add_node(Convolution2D(64, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv2_4', input='relu2_3')
graph.add_node(BatchNormalization(), name='bn2_4', input='conv2_4')
graph.add_node(Activation('relu'), name='relu2_4', inputs=['relu2_2','bn2_4'], 
               merge_mode='sum')
# Output shape = (64,56,56)

graph.add_node(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same', subsample=(2,2)),
               name='conv3_1', input='relu2_4')
graph.add_node(BatchNormalization(), name='bn3_1', input='conv3_1')
graph.add_node(Activation('relu'), name='relu3_1', input='bn3_1')
graph.add_node(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv3_2', input='relu3_1')
graph.add_node(BatchNormalization(), name='bn3_2', input='conv3_2')
# Output shape = (128,28,28)

graph.add_node(Convolution2D(32,1,1, W_regularizer=l2(weight_decay)),
               name='short3_1', input='relu2_2')
graph.add_node(BatchNormalization(), name='short3_2', input='short3_1')
# Output tensor shape = (32,56,56)
graph.add_node(Reshape((128,28,28)), name='short3_3', input='short3_2')               
graph.add_node(Activation('relu'), name='relu3_2', inputs=['short3_3','bn3_2'], 
               merge_mode='sum')
# Output shape = (128,28,28)

graph.add_node(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv3_3', input='relu3_2')
graph.add_node(BatchNormalization(), name='bn3_3', input='conv3_3')
graph.add_node(Activation('relu'), name='relu3_3', input='bn3_3')
graph.add_node(Convolution2D(128, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv3_4', input='relu3_3')
graph.add_node(BatchNormalization(), name='bn3_4', input='conv3_4')
graph.add_node(Activation('relu'), name='relu3_4', inputs=['relu3_2','bn3_4'], 
               merge_mode='sum')
# Output shape = (128,28,28)

graph.add_node(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same', subsample=(2,2)),
               name='conv4_1', input='relu3_4')
graph.add_node(BatchNormalization(), name='bn4_1', input='conv4_1')
graph.add_node(Activation('relu'), name='relu4_1', input='bn4_1')
graph.add_node(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv4_2', input='relu4_1')
graph.add_node(BatchNormalization(), name='bn4_2', input='conv4_2')
# Output shape = (256,14,14)

graph.add_node(Convolution2D(64,1,1, W_regularizer=l2(weight_decay)),
               name='short4_1', input='relu3_2')
graph.add_node(BatchNormalization(), name='short4_2', input='short4_1')
# Output shape = (64,28,28)
graph.add_node(Reshape((256,14,14)), name='short4_3', input='short4_2')               
graph.add_node(Activation('relu'), name='relu4_2', inputs=['short4_3','bn4_2'], 
               merge_mode='sum')
# Output shape = (256,14,14)


graph.add_node(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv4_3', input='relu4_2')
graph.add_node(BatchNormalization(), name='bn4_3', input='conv4_3')
graph.add_node(Activation('relu'), name='relu4_3', input='bn4_3')
graph.add_node(Convolution2D(256, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv4_4', input='relu4_3')
graph.add_node(BatchNormalization(), name='bn4_4', input='conv4_4')
graph.add_node(Activation('relu'), name='relu4_4', inputs=['relu4_2','bn4_4'], 
               merge_mode='sum')
# Output shape = (256,14,14)


graph.add_node(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same', subsample=(2,2)),
               name='conv5_1', input='relu4_4')
graph.add_node(BatchNormalization(), name='bn5_1', input='conv5_1')
graph.add_node(Activation('relu'), name='relu5_1', input='bn5_1')
graph.add_node(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv5_2', input='relu5_1')
graph.add_node(BatchNormalization(), name='bn5_2', input='conv5_2')
# Output shape = (256,14,14)

graph.add_node(Convolution2D(128,1,1, W_regularizer=l2(weight_decay)),
               name='short5_1', input='relu4_2')
graph.add_node(BatchNormalization(), name='short5_2', input='short5_1')
# Output shape = (64,28,28)
graph.add_node(Reshape((512,7,7)), name='short5_3', input='short5_2')               
graph.add_node(Activation('relu'), name='relu5_2', inputs=['short5_3','bn5_2'], 
               merge_mode='sum')
# Output shape = (512,7,7)
               
graph.add_node(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv5_3', input='relu5_2')
graph.add_node(BatchNormalization(), name='bn5_3', input='conv5_3')
graph.add_node(Activation('relu'), name='relu5_3', input='bn5_3')
graph.add_node(Convolution2D(512, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv5_4', input='relu5_3')
graph.add_node(BatchNormalization(), name='bn5_4', input='conv5_4')
graph.add_node(Activation('relu'), name='relu5_4', inputs=['relu5_2','bn5_4'], 
               merge_mode='sum')
# Output shape = (512,7,7)
                      
graph.add_node(AveragePooling2D(pool_size=(3,3), strides=(2,2),
                                border_mode='same'),
               name='pool2', input='relu5_4')               
graph.add_node(Flatten(), name='flatten', input='pool2')
graph.add_node(Dense(9, activation='sigmoid'), name='dense', input='flatten')
graph.add_output(name='output', input='dense')
               
               
sgd = SGD(lr=0.1, decay=1e-2, momentum=0.9)
graph.compile(optimizer=sgd, loss={'output':'binary_crossentropy'})


#%% Fit model
graph.fit({'input': tensor[train_ind],
           'output': train_df.iloc[train_ind,label_start:].values}, 
          batch_size=32, nb_epoch=number_of_epochs,
          validation_data={'input': tensor[test_ind],
                            'output': train_df.iloc[test_ind,label_start:].values},
          shuffle=True,
          callbacks=[EarlyStopping(monitor='val_loss', patience=0, mode='min')],
          verbose=1)


# Threshold at 0.5 and convert to 0 or 1 
predictions = (graph.predict({'input':tensor[test_ind]})['output'] > .5)*1

    
#%% Save model
if save_model:
    json_string = model.to_json()
    open(models_dir + model_name + '.json', 'w').write(json_string)
    model.save_weights(models_dir + model_name + '.h5')  # requires h5py
    
#%% Plot a few images to get a feel of how I did
if show_plots:
    for i in range(10):
        show_image_labels(tensor[test_ind[i]], predictions[i], 
                          train_df['labels'][test_ind[i]], im_mean)

