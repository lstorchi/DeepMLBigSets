import os
import re
import sys
import random
import argparse
import numpy as np

from tensorflow.keras.utils import to_categorical

import sys
sys.path.append("./common")

import commonutils

#####################################################################################3

if __name__ == "__main__":

    path = "./dbs/BBB_full/"
    npzsplit = "_c0_"

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", help="Specify the path where files are placed: " +  path , \
        type=str, required=False, default=path, dest="inputpath")
    parser.add_argument("--labelfile", help="Read labels file", \
        type=str, required=True)
    parser.add_argument("--splitter", help="Specify npz filename splitter, default: " + npzsplit , \
        type=str, required=False, default=npzsplit)

    args = parser.parse_args()

    path = args.inputpath
    npzsplit = args.splitter

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
                exit(1)

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

    yl_train = []
    yl_val = []
    yl_test = []

    for idx, files in enumerate(os.listdir(path)):
        if files.endswith(ext):

            name = files.split(".")[0]
            basename = re.split(npzsplit, name)[0]
            
            if basename in trainvalid_labels:
                rnd = random.uniform(0.0,1.0)

                if rnd > validsetdim:
                    train_names.append(path+"/"+files)
                    yl_train.append(labels[basename])
                else:
                    val_names.append(path+"/"+files)
                    yl_val.append(labels[basename])
            elif basename in test_labels:
                test_names.append(path+"/"+files)
                yl_test.append(labels[basename])
            else:
                print(name," and match ", basename)
                print("    Not in Set")
                sys.stdout.flush()

    print("Done...", flush=True)

    #labels_train = to_categorical(yl_train)
    #labels_val = to_categorical(yl_val)
    #labels_test = to_categorical(yl_test)

    labels_train = np.array(yl_train)
    labels_val = np.array(yl_val)
    labels_test = np.array(yl_test)

    print("Training: ", len(train_names), labels_train.shape)
    np.save("training_filenames.npy", train_names)
    np.save("training_labels.npy", labels_train)

    print("Validation: ", len(val_names), labels_val.shape)
    np.save("validation_filenames.npy", val_names)
    np.save("validation_labels.npy", labels_val)

    print("Test: ", len(test_names), labels_test.shape)
    np.save("test_filenames.npy", test_names)
    np.save("test_labels.npy", labels_test)
