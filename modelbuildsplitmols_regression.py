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

import matplotlib.pyplot as plt

#from keras import backend as K

import tensorflow as tf
#tf.config.experimental.set_lms_enabled(True)

from  tensorflow.keras import backend as K 

import sys
sys.path.append("./common")

import commonutils
import models

#import visualkeras
#from PIL import ImageFont

#####################################################################################3

def closest(lst, K):
      
    return lst[min(range(len(lst)), key = lambda i: math.fabs(lst[i]-K))]

#####################################################################################3

def  extract_predicions_mixedback (labels, predictions):

    labels_set = set()
    for v in labels:
        labels_set.add(v)

    samelabel_pediction = {}

    for v in labels_set:
        samelabel_pediction[v] = []
    
    for i, v in enumerate(labels):
        samelabel_pediction[v].append(predictions[i])

    best_prediction = []
    avg_pediction = []
    std_pediction = []
    single_vals = []

    best_mse = 0.0
    avg_mse = 0.0
    n = 0
    for realval in samelabel_pediction:
        n += 1
        single_vals.append(realval)

        cval = closest(samelabel_pediction[realval], realval)
        best_prediction.append(cval)
        best_mse += math.pow(cval - realval, 2.0)
        
        aval = np.mean(samelabel_pediction[realval])
        avg_pediction.append(aval)
        avg_mse += math.pow(aval - realval, 2.0)

        std_pediction.append(np.std(samelabel_pediction[realval]))

    best_mse = best_mse/float(n)
    avg_mse = avg_mse/float(n)

    return best_prediction, avg_pediction, std_pediction, single_vals, best_mse, avg_mse

#####################################################################################3

def extract_predicions (mol_to_idxvp, labels, predictions, idxvp_to_molname):

    moltodiff = {}

    best_prediction = []
    avg_pediction = []
    std_pediction = []
    single_vals = []
    best_mse = 0.0
    avg_mse = 0.0
    n = 0
    for mol in mol_to_idxvp:
        #print(mol) 
        vals = []
        checkvals = set()
        vpsvals = {}
        for i in mol_to_idxvp[mol]:
            molname = idxvp_to_molname[i]
            basename = re.split("_c\d+_", molname)[0]
            vpval = re.split("_c\d+_", molname)[1]
            vpname = re.split("\d+$", basename+"_"+vpval)[0]

            if not vpname in vpsvals:
                vpsvals[vpname] = []
            
            vpsvals[vpname].append(predictions[i])
            checkvals.add(labels[i])
            vals.append(predictions[i])

        if len(checkvals) > 1:
            print("Error non unique values")
            exit(-1)

        realval = checkvals.pop()

        single_vals.append(realval)

        n += 1
        cval = closest(vals, realval)
        best_mse += math.pow(cval - realval, 2.0)
        best_prediction.append(cval)

        moltodiff[mol] = math.fabs(cval - realval)

        l_avg_mse = 0.0
        l_n = 0.0
        for c in vpsvals:
            l_avg_mse += math.pow( (np.mean(vpsvals[c]) - realval), 2.0)
            l_n += 1.0
        maval = l_avg_mse/l_n
        aval = np.mean(vals)

        avg_mse += maval
        avg_pediction.append(aval)

        std_pediction.append(np.std(vals))

    print("Extracted ", n , " values ")

    best_mse = best_mse/float(n)
    avg_mse = avg_mse/float(n)

    return best_prediction, avg_pediction, std_pediction, \
        single_vals, best_mse, avg_mse, moltodiff

#####################################################################################3

if __name__ == "__main__":

    path = "./dbs/BBB_full/"
    cn = 8
    infilters = 32
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

    parser.add_argument("--filters", \
        help="Specify the number of output filters in the convolution, default: " + str(infilters)  , \
        type=int, required=False, default=infilters)
    parser.add_argument("--nunits", \
        help="Specify the dimensionality of the output space for the iner dense layers, default: " + str(infilters)  , \
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
    parser.add_argument("--removecn", help="Specify the list of channels to remove, ; separated string default: " + cntorm , \
        type=str, required=False, default=cntorm)
    parser.add_argument("--dumppredictions", help="Dump predicted data in CSV files", \
        action='store_true', default=False)

    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    path = args.inputpath
    cn = args.channels
    infilters = args.filters
    inunits = args.nunits
    indense_layers = args.ndenselayers
    nepochs = args.nepochs
    nbatch_size = args.nbatchsize
    modelname = args.modelname
    npzsplit = args.splitter
    cntorm = args.removecn

    cnformodel = cn

    mixuptrainanadvalback = True

    nulltestset = False
    trainsetdim = 0.60
    testsetdim = 0.50
    if nulltestset:
        testsetdim = 0.0

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
                val = float(sline[1])
                
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

    train_size = int(trainsetdim * len(labels))
    fullsize = len(labels)
    train_keys = random.sample(list(labels), train_size)

    train_labels = {}
    testval_labels = {}
    for k in labels:
        if k in train_keys:
            train_labels[k] = labels[k]
        else:
            testval_labels[k] = labels[k]

    test_size = int(testsetdim * len(testval_labels))
    test_keys = random.sample(list(testval_labels), test_size)

    test_labels = {}
    val_labels = {}
    for k in testval_labels:
        if k in test_keys:
            test_labels[k] = testval_labels[k]
        else:
            val_labels[k] = testval_labels[k]

    print("Training Molecules: ", len(train_labels), flush=True)
    print("Validation Molecules: ", len(val_labels), flush=True)
    print("Test Molecules: ", len(test_labels), flush=True)

    if ((len(train_labels) + len(val_labels) + len(test_labels)) != len(labels)):
        print("Error in diemsnion of sets")
        exit(1)

    # in case to check set
    # print(list(train_labels.items())[1])
    # print(list(train_labels.items())[2])
    # print(list(val_labels.items())[3])
    # print(list(val_labels.items())[6])

    counter = 1;
    for i, j in train_labels.items():
        print("InitialTrainset ", counter, i, j, flush=True)
        counter += 1

    for i, j in val_labels.items():
        print("InitialValset   ", counter, i, j, flush=True)
        counter += 1

    for i, j in test_labels.items():
        print("InitialTestset  ", counter, i, j, flush=True)
        counter += 1

    dicts = [train_labels, test_labels, val_labels]
    common_keys = set(train_labels.keys())
    for d in dicts[1:]:
        common_keys.intersection_update(set(d.keys()))
    
    if len(common_keys) != 0:
        print("Common keys check the split")
        exit(1)

    ext = ".npz" # devo provare a  caricarli tutti 
    
    yl_train = []
    Xl_train = []
    train_mol_to_idxvp = {}
    train_idxvp_to_molname = {}
    train_molnames = []

    yl_val = []
    Xl_val = []
    val_mol_to_idxvp = {}
    val_idxvp_to_molname = {}
    val_molnames = []

    yl_test = []
    Xl_test = []
    test_mol_to_idxvp = {}
    test_idxvp_to_molname = {}
    test_molnames = []
    
    glob_dimx = 0
    glob_dimy = 0
    glob_dimy = 0
    
    first = True

    cnformodel = cn 
    tormlist = []
    if cntorm != "" :
        for cnrm in cntorm.split(";"):
            print("I will remove channel :", cnrm)
            tormlist.append(int(cnrm))
            cnformodel -= 1

    print("Start reading files...", flush=True)

    npzfilelist = set()
    for files in os.listdir(path):
        if files.endswith(ext):
            name = files.split(".")[0]
            basename = re.split(npzsplit, name)[0]
            npzfilelist.add(basename)

    atleatone = False
    for file in listnames:
        if not file in npzfilelist:
            print("Error file ", file , " not present in NPZ")
            atleatone = True

    for file in npzfilelist:
        if not file in listnames:
            print("Error file ", file , " not present in data file")
            atleatone = True

    if atleatone:
        exit(1)

    for idx, files in enumerate(os.listdir(path)):
        if files.endswith(ext):

            name = files.split(".")[0]

            treedobject, dimx, dimy, dimz = \
                commonutils.readfeature (path, files, cn)

            if cntorm != "" and dimx != None:
                #print("initial treedobject Shape", treedobject.shape)
                treedobject = np.delete(treedobject, tormlist, 3)
                #print("treedobject Shape", treedobject.shape)

            if dimx != None:
                if first:
                    glob_dimx = dimx
                    glob_dimy = dimy
                    glob_dimz = dimz
                    first = False
                else:
                    if dimx != glob_dimx or \
                        dimy != glob_dimy or \
                            dimz != glob_dimz:
                        print(files, file= sys.stderr)
                        print("Dimension problem was ", glob_dimx, glob_dimy, glob_dimz, \
                            file=sys.stderr)
                        print("                  now ", dimx, dimy, dimz, \
                            file=sys.stderr)
                    else:
                        #basename = name.split(npzsplit)[0]
                        basename = re.split(npzsplit, name)[0]
                        #print("Reading: ", name," and match ", basename)
                        #print("  Dimensions: ",dimx, dimy, dimz)
                        
                        if basename in train_labels:
                            yl_train.append(labels[basename])
                            Xl_train.append(treedobject)
                            train_molnames.append(name)

                            if not basename in train_mol_to_idxvp:
                                train_mol_to_idxvp[basename] = []
                            train_mol_to_idxvp[basename].append(len(yl_train)-1)
                            train_idxvp_to_molname[len(yl_train)-1] = name
                            #print ("    In Training")
                        elif basename in val_labels:
                            yl_val.append(labels[basename])
                            Xl_val.append(treedobject)
                            val_molnames.append(name)

                            if not basename in val_mol_to_idxvp:
                                val_mol_to_idxvp[basename] = []
                            val_mol_to_idxvp[basename].append(len(yl_val)-1)
                            val_idxvp_to_molname[len(yl_val)-1] = name
                            #print ("    In Validation")
                        elif basename in test_labels:
                            yl_test.append(labels[basename])
                            Xl_test.append(treedobject)
                            test_molnames.append(name)

                            if not basename in test_mol_to_idxvp:
                                test_mol_to_idxvp[basename] = []
                            test_mol_to_idxvp[basename].append(len(yl_test)-1)
                            test_idxvp_to_molname[len(yl_test)-1] = name
                            #print ("    In Test")
                        else:
                            print("Reading: ", name," and match ", basename)
                            print("  Dimensions: ",dimx, dimy, dimz)
                            print("    Not in Set")
                            sys.stdout.flush()
            else:
                print("    Error in reading", name)

    print("Done...", flush=True)

    X_train = None
    y_train = None
    X_val = None
    y_val = None

    X_test = np.array(Xl_test, dtype=np.float32)
    y_test = np.array(yl_test, dtype=np.float32)

    if mixuptrainanadvalback:
        Xl_train.extend (Xl_val)
        yl_train.extend (yl_val)

        Xfull = np.array(Xl_train, dtype=np.float32)
        yfull = np.array(yl_train, dtype=np.float32)

        tests = testsetdim*(1 - trainsetdim)

        full_molnames = []

        full_molnames.extend(train_molnames)
        full_molnames.extend(val_molnames)

        #print("Get back moltoidx dict")

        X_train, X_val, y_train, y_val, train_molnames, val_molnames \
            = train_test_split(Xfull, yfull, full_molnames, \
            test_size=0.33, random_state=42)

        #for i in range(len(train_molnames)):
        #    print(train_molnames[i], y_train[i])

        train_mol_to_idxvp.clear()
        train_idxvp_to_molname.clear()

        counter = 0

        for i, name in enumerate(train_molnames):
            counter += 1

            confname = re.split("_vp\d+", name)[0]
            basename = re.split(npzsplit, name)[0]

            #print(name, confname, basename)

            val = labels[basename]
            print("Trainset %6d %s %11.6f"%(counter, confname, val))
            if not basename in train_mol_to_idxvp:
                train_mol_to_idxvp[basename] = []
            train_mol_to_idxvp[basename].append(i)
            train_idxvp_to_molname[i] = name
            #print(molname, basename, confname)

        #for n in train_mol_to_idxvp:
        #    print(n , " ==> ",train_mol_to_idxvp[n])

        val_mol_to_idxvp.clear()
        val_idxvp_to_molname.clear()

        for i, name in enumerate(val_molnames):
            counter += 1

            confname = re.split("_vp\d+", name)[0]
            basename = re.split(npzsplit, name)[0]
            val = labels[basename]
            print("Valset   %6d %s %11.6f"%(counter, confname, val))
            if not basename in val_mol_to_idxvp:
                val_mol_to_idxvp[basename] = []
            val_mol_to_idxvp[basename].append(i)
            val_idxvp_to_molname[i] = name
            #print(molname, basename, confname)

        for name in test_molnames:
            counter += 1
            basename = re.split(npzsplit, name)[0]
            val = labels[basename]
            print("Testset  %6d %s %11.6f"%(counter, name, val))
    else:
        X_train = np.array(Xl_train, dtype=np.float32)
        y_train = np.array(yl_train, dtype=np.float32)
    
        X_val = np.array(Xl_val, dtype=np.float32)
        y_val = np.array(yl_val, dtype=np.float32)

    print("Train: ", X_train.shape, y_train.shape)
    print("        ", len(train_molnames), len(train_mol_to_idxvp))
    print("Val:   ", X_val.shape, y_val.shape)
    print("        ", len(val_molnames), len(val_mol_to_idxvp))
    print("Test:  ", X_test.shape, y_test.shape)
    print("        ", len(test_molnames), len(test_mol_to_idxvp))

    #for i in test_idxvp_to_molname:
    #    print(i, test_idxvp_to_molname[i])

    #for b in test_mol_to_idxvp:
    #     print(b, test_mol_to_idxvp[b])

    #for i in val_idxvp_to_molname:
    #    print(i, val_idxvp_to_molname[i])

    #for b in val_mol_to_idxvp:
    #   print(b, val_mol_to_idxvp[b])

    labels_train = y_train
    labels_val = y_val
    labels_test = y_test

    #for n in train_mol_to_idxvp:
    #    print(n , " ==> ",train_mol_to_idxvp[n])
    #    for i in train_mol_to_idxvp[n]:
    #        print("    ", y_train[i])

    sample_shape = (glob_dimx, glob_dimy, glob_dimz, cnformodel)

    print("Sample shape: ", sample_shape)
   
    #model = models.model_scirep_regression(sample_shape, \
    #    indense_layers, inunits, infilters)

    model = models.model_scirep_regression_hyperopt(sample_shape, \
        indense_layers, inunits, [32,32,32], (3,3,3), (2, 2, 2), True)

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
    print ("Epoch MSE ValMSE")
    for i in range(len(history.history['mse'])):
        print ("%3d %12.8f %12.8f"%(i+1, history.history['mse'][i], 
            history.history['val_mse'][i]))

    plt.title('Keras model loss MAE')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    #plt.show()
    plt.savefig(modelname+'_loss.png')

    plt.clf()
    plt.title('Keras model MSE')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.legend(['training', 'validation'], loc='upper right')
    #plt.show()
    plt.savefig(modelname + '_mse.png')

    if not nulltestset: 

        score, tmse, tmae = model.evaluate(X_test, labels_test,
                            batch_size=nbatch_size)
        print('Test score:', score)
        print('Test mse:', tmse)
        print('Test mae:', tmae)

        test_predictions = model.predict(X_test)

        if args.dumppredictions:
            fp = open(modelname+"_testset_precictions.csv", "w")

            fp.write("Molname , Predicted , Truevalue\n")
            for i, pv in enumerate(test_predictions):
                name = test_molnames[i]
                fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, labels_test[i]))

            fp.close()

        msetest = 0.0
        for i in range(len(test_predictions)):
            msetest = msetest +  math.pow(test_predictions[i] - labels_test[i], 2.0)
        msetest = msetest / float(len(test_predictions))
        print ("Computed MSE test: %10.6f"%msetest)

        #for i in  range(predictions.shape[0]):
        #    print(labels_test[i], " vs ", predictions[i])

        best_test_predictions, avg_test_predictions, std_test_predictions, test_singlevals , \
            best_mse, avg_mse, moltodiff = \
            extract_predicions (test_mol_to_idxvp, labels_test, test_predictions, \
                test_idxvp_to_molname)

        plt.clf()
        plt.title("scatterplot for the testset")
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.xlim([-2.50, 2.50])
        plt.ylim([-2.50, 2.50])
        plt.scatter(labels_test, test_predictions)
        plt.savefig(modelname + '_scattertest.png')

        plt.clf()
        plt.title("scatterplot for the testset bestvalue")
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.xlim([-2.50, 2.50])
        plt.ylim([-2.50, 2.50]) 
        plt.scatter(test_singlevals, best_test_predictions)
        plt.savefig(modelname + '_scattertest_best.png')
       
        plt.clf()
        plt.title("scatterplot for the testset avgvalue")
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.xlim([-2.50, 2.50])
        plt.ylim([-2.50, 2.50]) 
        plt.scatter(test_singlevals, avg_test_predictions)
        plt.errorbar(test_singlevals, avg_test_predictions, yerr=std_test_predictions, fmt="o")
        plt.savefig(modelname + '_scattertest_avg.png')

        print("Test set, Best MSE: ", best_mse, " Avg MSE: ", avg_mse )

        sdiff = dict(sorted(moltodiff.items(), key=lambda item: item[1]))
        for mol in sdiff:
            print(mol, " ABS diff: ", sdiff[mol])


    train_predictions = model.predict(X_train)

    if args.dumppredictions:
        fp = open(modelname+"_trainingset_precictions.csv", "w")

        fp.write("Molname ; Predicted ; Truevalue\n")
        for i, pv in enumerate(train_predictions):
            name = train_molnames[i]
            fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, labels_train[i]))

        fp.close()

    msetrain = 0.0
    for i in range(len(train_predictions)):
        msetrain = msetrain +  math.pow(train_predictions[i] - labels_train[i], 2.0)
    msetrain = msetrain / float(len(train_predictions))
    print ("Computed MSE train: %10.6f"%msetrain)

    plt.clf()
    plt.title("scatterplot for the trainingset")
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([-2.50, 2.50])
    plt.ylim([-2.50, 2.50]) 
    plt.scatter(labels_train, train_predictions)
    plt.savefig(modelname + '_scattertrain.png')

    best_train_predictions = None 
    avg_train_predictions = None 
    std_train_predictions = None 
    train_singlevals = None 
    best_mse = None 
    avg_mse = None

    if not mixuptrainanadvalback:
        best_train_predictions, avg_train_predictions, std_train_predictions, train_singlevals, \
            best_mse, avg_mse, moltodiff = \
            extract_predicions (train_mol_to_idxvp, labels_train, train_predictions, \
                train_idxvp_to_molname)
    else:
        best_train_predictions, avg_train_predictions, std_train_predictions, train_singlevals, \
            best_mse, avg_mse = \
            extract_predicions_mixedback (labels_train, train_predictions)
       
    plt.clf()
    plt.title("scatterplot for the trainingset bestvalue")
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([-2.50, 2.50])
    plt.ylim([-2.50, 2.50])  
    plt.scatter(train_singlevals, best_train_predictions)
    plt.savefig(modelname + '_scattertrain_best.png')
    
    plt.clf()
    plt.title("scatterplot for the trainingset avgvalue")
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([-2.50, 2.50])
    plt.ylim([-2.50, 2.50])   
    plt.scatter(train_singlevals, avg_train_predictions)
    plt.errorbar(train_singlevals, avg_train_predictions, yerr=std_train_predictions, fmt="o")
    plt.savefig(modelname + '_scattertrain_avg.png')
    
    print("Training set, Best MSE: ", best_mse, " Avg MSE: ", avg_mse )

    val_predictions = model.predict(X_val)

    if args.dumppredictions:
        fp = open(modelname+"_validationset_precictions.csv", "w")
        
        fp.write("Molname , Predicted , Truevalue\n")
        for i, pv in enumerate(val_predictions):
            name = val_molnames[i]
            fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, labels_val[i]))

        fp.close()

    mseval = 0.0
    for i in range(len(val_predictions)):
        mseval = mseval +  math.pow(val_predictions[i] - labels_val[i], 2.0)
    mseval = mseval / float(len(val_predictions))
    print ("Computed MSE val: %10.6f"%mseval)
 
    plt.clf()
    plt.title("scatterplot for the validationset")
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([-2.50, 2.50])
    plt.ylim([-2.50, 2.50])  
    plt.scatter(labels_val, val_predictions)
    plt.savefig(modelname + '_scatterval.png')

    best_val_predictions = None 
    avg_val_predictions = None 
    std_val_predictions = None 
    val_singlevals = None

    if not mixuptrainanadvalback: 
        best_val_predictions, avg_val_predictions, std_val_predictions, val_singlevals, \
            best_mse, avg_mse, moltodiff = \
            extract_predicions (val_mol_to_idxvp, labels_val, val_predictions, \
                val_idxvp_to_molname)
    else:
        best_val_predictions, avg_val_predictions, std_val_predictions, val_singlevals, \
            best_mse, avg_mse = \
            extract_predicions_mixedback (labels_val, val_predictions)
       
    plt.clf()
    plt.title("scatterplot for the valset bestvalue")
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([-2.50, 2.50])
    plt.ylim([-2.50, 2.50])   
    plt.scatter(val_singlevals, best_val_predictions)
    plt.savefig(modelname + '_scatterval_best.png')
    
    plt.clf()
    plt.title("scatterplot for the valset avgvalue")
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([-2.50, 2.50])
    plt.ylim([-2.50, 2.50])    
    plt.scatter(val_singlevals, avg_val_predictions)
    plt.errorbar(val_singlevals, avg_val_predictions, yerr=std_val_predictions, fmt="o")
    plt.savefig(modelname + '_scatterval_avg.png')

    print("Validation set, Best MSE: ", best_mse, " Avg MSE: ", avg_mse )

    model.save(modelname)
