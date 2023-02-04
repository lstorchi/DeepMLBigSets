import os
import re
import sys
import math
import random
import argparse
import numpy as np

from socket import NI_NAMEREQD

import matplotlib.pyplot as plt

import tensorflow as tf
#tf.config.experimental.set_lms_enabled(True)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import sys
sys.path.append("./common")

import commonutils
import models

#import visualkeras
#from PIL import ImageFont



#####################################################################################3

if __name__ == "__main__":

    cn = 8
    modelname = "model"
 
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainfilenames", help="Specify the input training filenames ", 
        type=str, required=True)
    parser.add_argument("--trainlabels", help="Read training labels file", \
        type=str, required=True)
    parser.add_argument("--validfilenames", help="Specify the input validation filenames ", 
        type=str, required=True)
    parser.add_argument("--validlabels", help="Read validation labels file", \
        type=str, required=True)
    parser.add_argument("--testfilenames", help="Specify the input test filenames ", 
        type=str, required=True)
    parser.add_argument("--testlabels", help="Read test labels file", \
        type=str, required=True)

    parser.add_argument("--modelname", help="Specify modelname, default: " + modelname  , \
        type=str, required=False, default=modelname)
    parser.add_argument("--channels", help="Specify channels to be used, default: " + str(cn)  , \
        type=int, required=False, default=cn)
    parser.add_argument("--dumppredictions", help="Dump predicted data in CSV files", \
        action='store_true', default=False)

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), flush=True)

    cn = args.channels
    modelname = args.modelname

    cnformodel = cn

    X_train_filenames = np.load(args.trainfilenames)
    X_val_filenames = np.load(args.validfilenames)
    X_test_filenames = np.load(args.testfilenames)

    labels_train = np.load(args.trainlabels)
    labels_val = np.load(args.validlabels)
    labels_test = np.load(args.testlabels)

    val_samples = X_val_filenames.shape[0]

    print("Training Samplse: ", X_train_filenames.shape[0], flush=True)
    print("Validation Samples: ", X_val_filenames.shape[0], flush=True)
    print("Test Samples: ", X_test_filenames.shape[0], flush=True)

    dimx = 0
    dimy = 0
    dimz = 0

    print("Training labels: ", len(labels_train), flush=True)
    print("Validation labels: ", len(labels_val), flush=True)
    print("Test labels: ", len(labels_test), flush=True)

    First = True
    Xtestl = []
    for fname in X_test_filenames:
        treedobject, tdimx, tdimy, tdimz = \
            commonutils.readfeature("", fname, cn)
        Xtestl.append(treedobject)

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

    X_test = np.array(Xtestl)

    sample_shape = (dimx, dimy, dimz, cnformodel)

    print("Sample shape: ", sample_shape, flush=True)

    model = tf.keras.models.load_model (args.modelname)

    model.summary()

    labels_pred = model.predict(X_test).ravel()

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels_test, labels_pred)
    auc_keras = auc(fpr_keras, tpr_keras)

    print("AUC : ", auc_keras)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig("roc.png")