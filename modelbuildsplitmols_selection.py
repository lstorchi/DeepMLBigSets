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
    parser.add_argument("--trainlabels", help="Read training labels file", \
        type=str, required=True)
    parser.add_argument("--validfilenames", help="Specify the input traininglidation filenames ", 
        type=str, required=True)
    parser.add_argument("--validlabels", help="Read validation labels file", \
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

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)

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
    labels_train = np.load(args.trainlabels)
    labels_val = np.load(args.validlabels)

    train_samples = X_train_filenames.shape[0]
    val_samples = X_val_filenames.shape[0]

    print("Training Samplse: ", train_samples, flush=True)
    print("Validation Samples: ", val_samples, flush=True)

    dimx = 0
    dimy = 0
    dimz = 0

    print("Training labels: ", len(labels_train), flush=True)
    print("Validation labels: ", len(labels_val), flush=True)

    First = True
    Xvall = []
    for fname in X_val_filenames:
        treedobject, tdimx, tdimy, tdimz = \
            commonutils.readfeature("", fname, cn)
        Xvall.append(treedobject)

        if First:
            dimx = tdimx
            dimy = tdimy
            dimz = tdimz 
            First = False
        else:
            if (tdimx  != dimx) or (tdimy != dimy) or \
                (tdimz != dimz):
                print("Error in Dimension ", fname)
                exit(1)

    Xtranl = []
    for fname in X_train_filenames:
        treedobject, tdimx, tdimy, tdimz = \
            commonutils.readfeature("", fname, cn)
        Xtranl.append(treedobject)

        if (tdimx  != dimx) or (tdimy != dimy) or \
           (tdimz != dimz):
            print("Error in Dimension ", fname)
            exit(1)
   
    X_train = np.array(Xtranl)
    X_val =  np.array(Xvall)

    sample_shape = (dimx, dimy, dimz, cnformodel)

    print("Sample shape: ", sample_shape, flush=True)

    model = models.model_scirep_selection_hyperopt(sample_shape, indense_layers, inunits, \
       filteryouse , kernesizetouse, poolsizetouse, usethreecnn)

    K.set_value(model.optimizer.learning_rate, 0.0001)
    print("Learning rate before second fit:", model.optimizer.learning_rate.numpy(),\
         flush=True)
    
    model.summary()

    history = model.fit(X_train, labels_train,
                        batch_size=nbatch_size,
                        epochs=nepochs,
                        verbose=1,
                        validation_data=(X_val, labels_val))

    print("")
    print ("Epoch Loss ValLoss")
    for i in range(len(history.history['loss'])):
        print ("%3d %12.8f %12.8f"%(i+1, history.history['loss'][i], 
            history.history['val_loss'][i]), flush=True)

    print("")
    print ("Epoch Accuracy")
    for i in range(len(history.history['accuracy'])):
        print ("%3d %12.8f %12.8f"%(i+1, history.history['accuracy'][i], 
            history.history['val_accuracy'][i]), flush=True)
   
    model.save(modelname)