from glob import glob
import os, sys, json, math
import numpy as np
from numpy.random import permutation, random
from shutil import copyfile
import bcolz

from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D 
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
K.set_image_data_format('channels_first')

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())


def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True ,batch_size=8, 
                class_mode='categorical', target_size=(224, 224)):
    return gen.flow_from_directory(path, shuffle=shuffle, batch_size=batch_size, class_mode=class_mode)


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    return batches.classes, val_batches.classes, onehot(batches.classes), onehot(val_batches.classes), batches.filenames, val_batches.filenames, test_batches.filenames


def get_data(path, target_size=(224, 224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])
    

def get_split_model(model, layer_type):
    if layer_type == Conv2D: 
        last_conv_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Conv2D][-1]
        split_layer_idx = last_conv_idx + 1
    elif layer_type == Flatten:
        flatten_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Flatten][0]
        split_layer_idx = flatten_idx
        
    conv_model = Sequential(model.layers[:split_layer_idx])
    for layer in conv_model.layer: layer.trainable = False
    conv_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    fc_model = Sequential(model.layers[split_layer_idx:])
    for layer in fc_model.layer: layer.trainable = True
    conv_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return conv_model, fc_model
    
    
def ceil(x):
    return int(math.ceil(x))


def floor(x):
    return int(math.floor(x))


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

    
def load_array(fname):
    return bcolz.open(rootdir=fname)

