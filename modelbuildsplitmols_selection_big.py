import os
import re
import sys
import math
import random
import argparse
import numpy as np

from socket import NI_NAMEREQD
from pandas import crosstab

from sklearn.model_selection import train_test_split

#from keras import backend as K

import tensorflow as tf
#tf.config.experimental.set_lms_enabled(True)

from tensorflow.keras import backend as K 
from tensorflow.keras.utils import to_categorical

import sys
sys.path.append("./common")

import commonutils
import models

#import visualkeras
#from PIL import ImageFont

#####################################################################################

class readergenerator(tf.keras.utils.Sequence):

  def __init__(self, filenames, labels, batch_size) :
    self.filenames = filenames
    self.labels = labels
    self.batch_size = batch_size

    self.dimx = 0
    self.dimy = 0
    self.dimz = 0
    
  def __len__(self) :
    return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(int)
  
  def __getitem__(self, idx) :
    batch_x = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    cn = 8
    X = []    
    for file_name in batch_x:
        treedobject, self.dimx, self.dimy, self.dimz = \
            commonutils.readfeature("", file_name, cn)
        X.append(treedobject)
        #print(file_name)

    return np.array(X), np.array(batch_y)

#####################################################################################3

if __name__ == "__main__":

    cn = 8
    inunits = 64
    indense_layers = 4
    nepochs = 30
    nbatch_size=64
    modelname = "model"
    npzsplit = "_c0_"
    filteryouse = [64,64,64]
    poolsizetouse = (2, 2, 2)
    kernesizetouse = (3,3,3)
    usethreecnn = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--trainfilenames", help="Specify the input training filenames ", 
        type=str, required=True)
    parser.add_argument("--trainlabels", help="Read train labels file", \
        type=str, required=True)
    parser.add_argument("--validfilenames", help="Specify the input training filenames ", 
        type=str, required=True)
    parser.add_argument("--validlabels", help="Read train labels file", \
        type=str, required=True)

    parser.add_argument("--nunits", \
        help="Specify the dimensionality of the output space for the iner dense layers, default: " + \
            str(inunits)  , \
        type=int, required=False, default=inunits)
    parser.add_argument("--ndenselayers", help="Specify the number of inner dense layers, default: " + \
        str(indense_layers)  , \
        type=int, required=False, default=indense_layers)
    parser.add_argument("--nbatchsize", help="Specify the batch size, default: " + str(nbatch_size)  , \
        type=int, required=False, default=nbatch_size)
    parser.add_argument("--nepochs", help="Specify the number of epochs, default: " + str(nepochs)  , \
        type=int, required=False, default=nepochs)
    parser.add_argument("--filters", \
        help="Specify filters to use as as list-like string dim 2 or 3 depends on usethreelayets, default: " + \
        str(filteryouse), type=str, required=False, default=str(filteryouse))
    parser.add_argument("--poolsize", \
        help="Specify Pool Size filters as tuple-like string, default: " + str(poolsizetouse)  , \
        type=str, required=False, default=str(poolsizetouse))
    parser.add_argument("--kernelsize", \
        help="Specify Kernel Size filters as tuple-like string, default: " + str(kernesizetouse)  , \
        type=str, required=False, default=str(kernesizetouse))
    parser.add_argument("--nocnnlayers3", help="Use 3 CNN layers", \
        action='store_true', default=False)

    parser.add_argument("--modelname", help="Specify modelname, default: " + modelname  , \
        type=str, required=False, default=modelname)
    parser.add_argument("--channels", help="Specify channels to be used, default: " + str(cn)  , \
        type=int, required=False, default=cn)
    parser.add_argument("--dumppredictions", help="Dump predicted data in CSV files", \
        action='store_true', default=False)

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    cn = args.channels
    inunits = args.nunits
    indense_layers = args.ndenselayers
    nepochs = args.nepochs
    nbatch_size = args.nbatchsize
    modelname = args.modelname
    filteryouse = eval(args.filters)
    poolsizetouse = eval(args.poolsize)
    kernesizetouse = eval(args.kernelsize)
    usethreecnn = not(args.nocnnlayers3)

    cnformodel = cn

    batch_size = 500
    train_samples = 0
    val_samples = 0

    X_train_filenames = np.load(args.trainfilenames)
    X_val_filenames = np.load(args.validfilenames)
    y_train = np.load(args.trainlabels)
    y_val = np.load(args.validlabels)

    train_samples = X_train_filenames.shape[0]
    val_samples = X_val_filenames.shape[0]

    print("Training Samplse: ", train_samples)
    print("Validation Samples: ", val_samples)

    # read firts fopr dimension
    dimx = 0
    dimy = 0
    dimz = 0
    name = X_train_filenames[0]
    treedobject, dimx, dimy, dimz = commonutils.readfeature("", name, cn)
    print("Reading first element dimensions: ", dimx, dimy, dimz)
 
    training_batch_generator = readergenerator(X_train_filenames, y_train, batch_size)
    validation_batch_generator = readergenerator(X_val_filenames, y_val, batch_size)

    sample_shape = (dimx, dimy, dimz, cnformodel)

    print("Sample shape: ", sample_shape)

    model = models.model_scirep_selection_hyperopt(sample_shape, indense_layers, inunits, \
       filteryouse , kernesizetouse, poolsizetouse, usethreecnn)

    K.set_value(model.optimizer.learning_rate, 0.0001)
    print("Learning rate before second fit:", model.optimizer.learning_rate.numpy())
    
    model.summary()

    history = model.fit(training_batch_generator,
                   steps_per_epoch = int(train_samples // batch_size), # instead of ceil
                   epochs = 10,
                   verbose = 1,
                   validation_data = validation_batch_generator,
                   validation_steps = int(val_samples // batch_size))


    print("")
    print ("Epoch Loss ValLoss")
    for i in range(len(history.history['loss'])):
        print ("%3d %12.8f %12.8f"%(i+1, history.history['loss'][i], 
            history.history['val_loss'][i]))

    print("")
    print ("Epoch Accuracy")
    for i in range(len(history.history['accuracy'])):
        print ("%3d %12.8f %12.8f"%(i+1, history.history['accuracy'][i], 
            history.history['val_accuracy'][i]))
   
    model.save(modelname)