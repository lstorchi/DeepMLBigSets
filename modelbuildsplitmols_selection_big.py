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

    path = "./dbs/BBB_full/"
    cn = 8
    inunits = 64
    indense_layers = 4
    nepochs = 30
    nbatch_size=64
    modelname = "model"
    npzsplit = "_c0_"
    cntorm = ""

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", help="Specify the path where files are placed: " +  path , \
        type=str, required=False, default=path, dest="inputpath")
    parser.add_argument("--labelfile", help="Read labels file", \
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
    parser.add_argument("--modelname", help="Specify modelname, default: " + modelname  , \
        type=str, required=False, default=modelname)
    parser.add_argument("--channels", help="Specify channels to be used, default: " + str(cn)  , \
        type=int, required=False, default=cn)
    parser.add_argument("--splitter", help="Specify npz filename splitter, default: " + npzsplit , \
        type=str, required=False, default=npzsplit)
    parser.add_argument("--dumppredictions", help="Dump predicted data in CSV files", \
        action='store_true', default=False)

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    path = args.inputpath
    cn = args.channels
    inunits = args.nunits
    indense_layers = args.ndenselayers
    nepochs = args.nepochs
    nbatch_size = args.nbatchsize
    modelname = args.modelname
    npzsplit = args.splitter

    cnformodel = cn

    labels = {}
    random.seed(1)

    fp = open(args.labelfile, "r")

    listnames = []

    for line in fp:
        sline = line.split()

        if len(sline) != 2:
            print("Error in line ", line, file=sys.stderr)
        else:
            if (commonutils.check_float(sline[1])):
                val = int(sline[1])
                
                if (sline[0] in labels):
                    print("Error filename ", sline[0], " already in")
                    print(line)

                labels[sline[0]] = val
                listnames.append(sline[0])
            else:
                print("Error in line ", line, "non numeric value ", file=sys.stderr)

    if len(listnames) != len(labels):
        print("duplicated file")
        exit(1)

    trainvalidsetdim = 0.80
    validsetdim = 0.20
    testsetdim = 0.20

    trainvalid_size = int(trainvalidsetdim * len(labels))
    fullsize = len(labels)
    train_keys = random.sample(list(labels), trainvalid_size)

    trainvalid_labels = {}
    test_labels = {}
    for k in labels:
        if k in train_keys:
            trainvalid_labels[k] = labels[k]
        else:
            test_labels[k] = labels[k]

    test_size = int(testsetdim * len(test_labels))
    test_keys = random.sample(list(test_labels), test_size)

    print("Training+Validation Molecules: ", len(trainvalid_labels), flush=True)
    print("Test Molecules: ", len(test_labels), flush=True)

    if ((len(trainvalid_labels) + len(test_labels)) != len(labels)):
        print("Error in diemsnion of sets")
        exit(1)

    # in case to check set
    # print(list(train_labels.items())[1])
    # print(list(train_labels.items())[2])
    # print(list(val_labels.items())[3])
    # print(list(val_labels.items())[6])

    counter = 1;
    for i, j in trainvalid_labels.items():
        print("InitialTrain+Validset ", counter, i, j, flush=True)
        counter += 1

    for i, j in test_labels.items():
        print("InitialTestset  ", counter, i, j, flush=True)
        counter += 1

    ext = ".npz" # devo provare a  caricarli tutti 
    
    train_names = []
    val_names = []
    test_names = []

    first = True

    glob_dimx = 0
    glob_dimy = 0
    glob_dimy = 0

    cnformodel = cn
    for idx, files in enumerate(os.listdir(path)):
        if files.endswith(ext):

            name = files.split(".")[0]

            if first:
                treedobject, glob_dimx, glob_dimy, glob_dimz = \
                    commonutils.readfeature (path, files, cn)
                first = False

            #basename = name.split(npzsplit)[0]
            basename = re.split(npzsplit, name)[0]
            #print("Reading: ", name," and match ", basename)
            #print("  Dimensions: ",dimx, dimy, dimz)
            
            if basename in trainvalid_labels:
                rnd = random.uniform(0.0,1.0)

                if rnd > validsetdim:
                    train_names.append(name)
                else:
                    val_names.append(name)
            elif basename in test_labels:
                test_names.append(name)
            else:
                print("Reading: ", name," and match ", basename)
                print("  Dimensions: ",dimx, dimy, dimz)
                print("    Not in Set")
                sys.stdout.flush()

    print("Start reading files...", flush=True)

    cnformodel = cn

    yl_train = []
    yl_val = []
    yl_test = []

    X_test = np.zeros((len(test_names), glob_dimx, glob_dimy, glob_dimz, cnformodel),\
         dtype=np.float32)
    X_train = np.zeros((len(train_names), glob_dimx, glob_dimy, glob_dimz, cnformodel), \
        dtype=np.float32)
    X_val = np.zeros((len(val_names), glob_dimx, glob_dimy, glob_dimz, cnformodel), \
        dtype=np.float32)

    trainidx = 0
    testidx = 0
    validx = 0

    for idx, files in enumerate(os.listdir(path)):
        if files.endswith(ext):

            name = files.split(".")[0]

            treedobject, dimx, dimy, dimz = \
                commonutils.readfeature (path, files, cn)

            if dimx != None:
                if dimx != glob_dimx or \
                    dimy != glob_dimy or \
                        dimz != glob_dimz:
                    print(files, file= sys.stderr)
                    print("Dimension problem was ", glob_dimx, glob_dimy, glob_dimz, \
                        file=sys.stderr)
                    print("                  now ", dimx, dimy, dimz, \
                        file=sys.stderr)
                    exit(1)
                else:
                    #basename = name.split(npzsplit)[0]
                    basename = re.split(npzsplit, name)[0]
                    #print("Reading: ", name," and match ", basename)
                    #print("  Dimensions: ",dimx, dimy, dimz)
                    
                    if name in train_names:
                        yl_train.append(labels[basename])
                        X_train[trainidx, :,  :, :, :] =  treedobject
                        trainidx += 1
                    elif name in val_names:
                        yl_val.append(labels[basename])
                        X_val[validx, :,  :, :, :] =  treedobject
                        validx *= 1
                    elif name in test_names:
                        yl_test.append(labels[basename])
                        X_test[testidx, :,  :, :, :] =  treedobject
                        testidx += 1
                    else:
                        print("Reading: ", name," and match ", basename)
                        print("  Dimensions: ",dimx, dimy, dimz)
                        print("    Not in Set")
                        sys.stdout.flush()
            else:
                print("    Error in reading", name)
                exit(1)

    print("Done...", flush=True)

    print("Train: ", X_train.shape, len(yl_train))
    print("Val:   ", X_val.shape, len(yl_val))
    print("Test:  ", X_test.shape, len(yl_test))

    labels_train = to_categorical(yl_train)
    labels_val = to_categorical(yl_val)
    labels_test = to_categorical(yl_test)

    #for n in train_mol_to_idxvp:
    #    print(n , " ==> ",train_mol_to_idxvp[n])
    #    for i in train_mol_to_idxvp[n]:
    #        print("    ", y_train[i])

    sample_shape = (glob_dimx, glob_dimy, glob_dimz, cnformodel)

    print("Sample shape: ", sample_shape)
   
    #model = models.model_scirep_selection(sample_shape, indense_layers, inunits, infilters)

    model = models.model_scirep_selection_hyperopt(sample_shape, indense_layers, inunits, \
        [32,32,32], (3,3,3), (2, 2, 2), True)

    #font = ImageFont.truetype("arial.ttf", 12)
    #visualkeras.layered_view(model, legend=True, font=font, draw_volume=False) 

    # lower learning rate the mse is stucked
    K.set_value(model.optimizer.learning_rate, 0.0001)
    print("Learning rate before second fit:", model.optimizer.learning_rate.numpy())
    
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
            history.history['val_loss'][i]))

    print("")
    print ("Epoch Accuracy")
    for i in range(len(history.history['accuracy'])):
        print ("%3d %12.8f %12.8f"%(i+1, history.history['accuracy'][i], 
            history.history['val_accuracy'][i]))
   
    model.save(modelname)

    fp = open("testset_"+modelname+".csv", "w")
    test_predict = model.predict(X_test)
    for i, name in enumerate(test_names):
        print(name, " , ", test_predict[i], " , ", labels_test[i], file=fp)
    fp.close()

    fp = open("trainset_"+modelname+".csv", "w")
    train_predict = model.predict(X_train)
    for i, name in enumerate(train_names):
        print(name, " , ", train_predict[i], " , ", labels_train[i], file=fp)
    fp.close()

    fp = open("validset_"+modelname+".csv", "w")
    val_predict = model.predict(X_val)
    for i, name in enumerate(val_names):
        print(name, " , ", val_predict[i], " , ", labels_val[i], file=fp)
    fp.close()
