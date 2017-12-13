from glob import glob
import os, sys, json, math
import numpy as np
from numpy.random import permutation, random
from shutil import copyfile
import bcolz

from sklearn.preprocessing import OneHotEncoder

import pandas as pd

from keras.utils import get_file, to_categorical
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D 
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
K.set_image_data_format('channels_first')

from keras.metrics import categorical_accuracy as accuracy
from keras.metrics import categorical_crossentropy as crossentropy


def onehot(x):
    return to_categorical(x)
    #return np.array(OneHotEncoder().fit_transform(x.reshape(-1, 1)).todense())


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
    return np.concatenate([batches.next() for i in range(batches.n)])
    

def get_split_models(model):
    """
        Gets the two models spliting convolution model and dense model at Flatten layer
            
        Returns:
        conv_model: the model constructing by Vgg convolution layers ending at the last MaxPooling2D layer 
        fc_model: the model constructing by Vgg dense layers starting at Flatten layer
            
    """
    flatten_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Flatten][0]
        
    conv_model = Sequential(model.layers[:flatten_idx])
    for layer in conv_model.layers: layer.trainable = False
    conv_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    fc_model = Sequential([ 
            Flatten(input_shape=conv_model.layers[-1].output_shape[1:]) 
        ])
    for layer in model.layers[flatten_idx+1:]: fc_model.add(layer)
    for layer in fc_model.layers: layer.trainable = True
    fc_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return conv_model, fc_model
    

def set_trainable(model, layer_type=Flatten):
    """
        Set specific type of layer in the model to be trainable
        
        Args:
            model: vgg model
            layer_type: Dense, Conv2D, ...etc
    """
    if layer_type == Conv2D: 
        last_conv_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Conv2D][-1]
        for layer in model.layers[:last_conv_idx+1]: layer.trainable = True
            
    elif layer_type == Flatten:
        flatten_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Flatten][0]
        for layer in model.layers[flatten_idx:]: layer.trainable = True
            
    elif layer_type == Dense:
        for layer in model.layers: 
            if type(layer) == Dense: layer.trainable = True

                
def eval_accuracy(labels, preds):
    """
        https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    """
   
    acc_value = accuracy(labels, preds)
    with sess.as_default():
        eval_result = acc_value.eval()
    return eval_result.mean()


def eval_crossentropy(labels, preds):
    """
        
        Ref: https://stackoverflow.com/questions/46687064/categorical-crossentropy-loss-no-attribute-get-shape
    """
   
    entropy_value = crossentropy(K.constant(labels.astype('float32')), K.constant(preds.astype('float32')))
    with sess.as_default():
        eval_result = entropy_value.eval()
    return eval_result.mean()


def do_clip(preds, max_pred): 
    """
       Finds a good clipping amount using the validation set, prior to submitting.
       
       Args:
           preds: prediction array for all prediction classes
           max_pred: maximum prediction clip edge ex: 0.90, 0.93, 0.95, 0.98...etc
       Returns:
           the prediction array after clipping by maximum and minimum clipping edge
    """
    return np.clip(preds, (1-max_pred)/9, max_pred)

        
def ceil(x):
    return int(math.ceil(x))


def floor(x):
    return int(math.floor(x))


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

    
def load_array(fname):
    return bcolz.open(rootdir=fname)


class MixIterator(object):
    
    def __init__(self, iters):
        self.iters = iters
        self.n = sum([itr.n for itr in self.iters])
        self.batch_size = sum([itr.batch_size for itr in self.iters])
        self.steps_per_epoch = sum([ceil(itr.n/itr.batch_size) for itr in self.iters])
    
    def reset(self):
        for itr in self.iters: itr.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self, *args, **kwargs):
        nexts = [next(itr) for itr in self.iters]
        n0 = np.concatenate([n[0] for n in nexts])
        n1 = np.concatenate([n[1] for n in nexts])
        return (n0, n1)

    
class PseudoLabelGenerator(object):
    
    def __init__(self, iterator, model):
        self.iter = iterator
        self.n = self.iter.n
        self.batch_size = self.iter.batch_size
        self.steps_per_epoch = ceil(self.iter.n/self.iter.batch_size)
        self.model = model
    
    def reset(self):
        self.iter.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self, *args, **kwargs):
        nexts = next(self.iter)
        results = self.model.predict(nexts[0], batch_size=self.batch_size)
        return (nexts[0], results)

