from glob import glob
import os, sys, json, math
import numpy as np
from numpy.random import permutation, random, choice
from shutil import copyfile
import bcolz

from PIL import Image

from sklearn.preprocessing import OneHotEncoder
from scipy.ndimage import *
from scipy.misc import *
from sklearn.metrics import confusion_matrix

import itertools
from itertools import chain

from matplotlib import pyplot as plt

import pandas as pd

from keras.utils import get_file, to_categorical
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D 
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
# K.set_image_data_format('channels_first')
K.set_image_data_format('channels_last')

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
    

def set_trainable(model, layer_type='Flatten'):
    """
        Set specific type of layer in the model to be trainable
        
        Args:
            model: vgg model
            layer_type: Dense, Conv2D, ...etc
    """
    if layer_type == 'Conv2D': 
        last_conv_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Conv2D][-1]
        for layer in model.layers[:last_conv_idx+1]: layer.trainable = True
            
    elif layer_type == 'Flatten':
        flatten_idx = [idx for idx, layer in enumerate(model.layers) if type(layer)==Flatten][0]
        for layer in model.layers[flatten_idx:]: layer.trainable = True
            
    elif layer_type == 'Dense':
        for layer in model.layers: 
            if type(layer) == Dense: layer.trainable = True
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    

def plot(img, title=None):
    if type(img) is np.ndarray:
        img = np.array(img).astype(np.uint8)
        # make sure ims is channel-last
        if img.shape[-1] != 3:
            img = img.transpose((1,2,0))
    f = plt.figure()
    sp = plt.subplot(111)
    sp.axis('off')
    sp.set_title(title, fontsize=16)
    plt.imshow(img)
                

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        # make sure ims is channel-last
        if ims.shape[-1] != 3:
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = int(math.ceil(len(ims)/rows))
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


def augment_view(param_name, choices, path, batch_size=64):
    batches = get_batches(path, shuffle=False, batch_size=batch_size)
    
    idx = int(np.random.choice(len(batches.filenames), 1)[0])
    # Create a 'batch' of a single image
    img_orig = imread(path+batches.filenames[idx])
    img_orig_batch = np.expand_dims(img_orig, axis=0)
    
    # original data
    plot(img_orig, title='original image')

    for c in choices:
        if param_name is 'rotation_range':
            gen = image.ImageDataGenerator(rotation_range=c, horizontal_flip=True, data_format='channels_last')
        elif param_name is 'width_shift_range':
            gen = image.ImageDataGenerator(width_shift_range=c, horizontal_flip=True, data_format='channels_last')
        elif param_name is 'height_shift_range':
            gen = image.ImageDataGenerator(height_shift_range=c, horizontal_flip=True, data_format='channels_last')
        elif param_name is 'shear_range':
            gen = image.ImageDataGenerator(shear_range=c, horizontal_flip=True, data_format='channels_last')
        elif param_name is 'zoom_range':
            gen = image.ImageDataGenerator(zoom_range=c, horizontal_flip=True, data_format='channels_last')
        elif param_name is 'channel_shift_range':
            gen = image.ImageDataGenerator(channel_shift_range=c, horizontal_flip=True, data_format='channels_last')
        else:
            return

        # Request the generator to create batches from this image
        aug_iter = gen.flow(img_orig_batch)

        # Get eight examples of these augmented images
        aug_imgs = [np.squeeze(next(aug_iter), axis=0).astype(np.uint8) for i in range(8)]

        # Augmented data
        plots(aug_imgs, (20,8), rows=1, titles=[param_name+' \n= '+str(c)+'\n augmented' for i in range(len(aug_imgs))])

                
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


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize=True'
        (This function is copied from scikit docs: https://github.com/scikit-learn/scikit-learn/blob/master/examples/model_selection/plot_confusion_matrix.py)
        
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
class MixIterator(object):
    
    def __init__(self, iters):
        self.iters = iters
        self.n = int(np.sum([itr.n for itr in self.iters]))
        self.batch_size = int(np.sum([itr.batch_size for itr in self.iters]))
        self.steps_per_epoch = max([ceil(itr.n/itr.batch_size) for itr in self.iters])
    
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
        self.class_indices = self.iter.class_indices
        #self.classes = np.argmax(self.model.predict_generator(self.iter,steps=self.steps_per_epoch), axis=1)
    
    def reset(self):
        self.iter.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self, *args, **kwargs):
        nexts = next(self.iter)
        results = self.model.predict(nexts[0], batch_size=self.batch_size)
        return (nexts[0], results)

