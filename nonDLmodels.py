import math
import re
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_val_score, train_test_split)

#from rdkit.Chem import PandasTools

import process_CSV_model_output as pcsv

#####################################################################################3

def closest(lst, K):
      
    return lst[min(range(len(lst)), key = lambda i: math.fabs(lst[i]-K))]

#####################################################################################3

def extract_predicions (mol_to_idxvp, labels, predictions):

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
        for i in mol_to_idxvp[mol]:
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

        aval = np.mean(vals)
        avg_mse += math.pow(aval - realval, 2.0)
        avg_pediction.append(aval)

        std_pediction.append(np.std(vals))

    print("Extracted ", n , " values ")

    best_mse = best_mse/float(n)
    avg_mse = avg_mse/float(n)

    return best_prediction, avg_pediction, std_pediction, \
        single_vals, best_mse, avg_mse, moltodiff

#####################################################################################3

def plot_scatter ( title, xlabel, ylabel, xdata, ydata, colour, alph, imgfile ):
    with pyplot.style.context('ggplot'):
        pyplot.clf()
        pyplot.figure(figsize=(10, 10))

        pyplot.rc('font', size=8)          # controls default text sizes
        pyplot.rc('axes', titlesize=6)     # fontsize of the axes title
        pyplot.rc('axes', labelsize=10)    # fontsize of the x and y labels
        pyplot.rc('xtick', labelsize=6)    # fontsize of the tick labels
        pyplot.rc('ytick', labelsize=6)    # fontsize of the tick labels
        pyplot.rc('legend', fontsize=10)    # legend fontsize
        pyplot.rc('figure', titlesize=12)  # fontsize of the figure title

        pyplot.title(title, fontsize = 12)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)
        pyplot.xlim([-2.50, 2.50])
        pyplot.ylim([-2.50, 2.50]) 
        pyplot.plot(xdata, ydata, colour, alpha=alph)
        fig1 = pyplot.gcf()
        fig1.set_size_inches(5, 4)
        fig1.savefig(imgfile, bbox_inches="tight", dpi=600)

        pyplot.close()

    return

#####################################################################################3

def plot_scatter_with_errors ( title, xlabel, ylabel, xdata, ydata, yerrors, pointfmt, barcolour, alph, imgfile ):
    with pyplot.style.context('ggplot'):
        pyplot.clf()
        pyplot.figure(figsize=(10, 10))

        pyplot.rc('font', size=8)          # controls default text sizes
        pyplot.rc('axes', titlesize=6)     # fontsize of the axes title
        pyplot.rc('axes', labelsize=10)    # fontsize of the x and y labels
        pyplot.rc('xtick', labelsize=6)    # fontsize of the tick labels
        pyplot.rc('ytick', labelsize=6)    # fontsize of the tick labels
        pyplot.rc('legend', fontsize=10)    # legend fontsize
        pyplot.rc('figure', titlesize=12)  # fontsize of the figure title
        pyplot.rc('errorbar', capsize=2)

        pyplot.title(title, fontsize = 12)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)
        pyplot.xlim([-2.50, 2.50])
        pyplot.ylim([-2.50, 2.50]) 

        markers, bars, caps = pyplot.errorbar(xdata, ydata, \
            yerr=yerrors, fmt=pointfmt, ecolor=barcolour, alpha=alph)
        [bar.set_alpha(alph) for bar in bars]
        [cap.set_alpha(alph) for cap in caps]

        fig1 = pyplot.gcf()
        fig1.set_size_inches(5, 4)
        fig1.savefig(imgfile, bbox_inches="tight", dpi=600)

        pyplot.close()
    return

#####################################################################################3

if __name__ == "__main__":

    QUICKTEST = False
    DROPCORRELATED = True
    optimizegmfe = True

    filename = ""
    #sdfile = ""
    traintestsetlist = ""
    
    if len (sys.argv) != 3:
        print("Error: usage ", sys.argv[0], " features.csv traintestsetlist.txt")
        exit(1)
    else:
        filename = sys.argv[1]
        traintestsetlist = sys.argv[2]
        #sdfile = sys.argv[3]
    
    df = pd.read_csv(filename, sep=";")
    
    df.drop('LgBB', inplace=True, axis=1)
    
    for torm in ["PB","VD","CACO2","SKIN", "CUSTOM"]:
        df.drop(torm, inplace=True, axis=1)
    
    if DROPCORRELATED:
        listtobeused = ["V","R","d","W1","W2","W5","W6","W7","W8","D1","D3","WO1","WO4","WO5","WO6","IW1","IW2","IW3","IW4","CW1","CW5","CW6","CW7","ID1","ID2","ID3","ID4","CD1","CD2","CD3","CD4","HL1","HL2","A","CP","FLEX","FLEX_RB","NCC","SE","SE0","SE1","LOGP n-Oct","LOGP c-Hex","PSA","PSAR","LgD5","AUS7.4","%FU4","%FU8","%FU9","%FU10","SOLY","LgS3","MpKaA","mpKaB","MetStab","HTSflag","C1","C2","C3","C4","C5","P1","P2","P3","P4","P5","EMDIF","EMDIS","FLEX_PT","PAINS"]
    
        todropdesc = []
        for desc in df.columns:
            if desc != "Objects":
                if desc not in listtobeused:
                    todropdesc.append(desc)
    
        for torm in todropdesc:
            df.drop(torm, inplace=True, axis=1)
    
    featuresusedlist = []
    print("Colums used: ")
    for i, c in enumerate(df.columns):
        print("%5d - %s"%(i+1, c))
        if (c != "Objects") and (c != "yval"):
            featuresusedlist.append(c)
    print()
    
    molecules = None
    
    X_train = None 
    y_train = None
    
    X_test = None
    y_test = None
    
    X_valid = None
    y_valid = None
    
    X_full = None
    y_full = None 
    
    #molecules = PandasTools.LoadSDF(sdfile,
    #                    smilesName='SMILES',
    #                    molColName='Molecule',
    #                    includeFingerprints=False)
    
    fp = open(traintestsetlist, "r")
    
    trainsetl = [] 
    testsetl = []
    validsetl = []
    
    moltolabels = {}
    
    npzsplit = "_c\d+"
    
    for line in fp:
        
        if line.startswith("InitialTrainset"):
            molname = line.split()[-2]
            label = float(line.split()[-1])
    
            trainsetl.append(molname)
    
            if molname in moltolabels:
                if label != moltolabels[molname]:
                    print("Error in ", molname)
                    exit(1)
            else:
                moltolabels[molname] = label
        elif line.startswith("InitialValset"):
            molname = line.split()[-2]
            label = float(line.split()[-1])
    
            validsetl.append(molname)
    
            if molname in moltolabels:
                if label != moltolabels[molname]:
                    print("Error in ", molname)
                    exit(1)
            else:
                moltolabels[molname] = label
        elif line.startswith("InitialTestset"):
            label = float(line.split()[-1])
            molname = line.split()[-2]
    
            testsetl.append(molname)
    
            if molname in moltolabels:
                if label != moltolabels[molname]:
                    print("Error in ", molname)
                    exit(1)
            else:
                moltolabels[molname] = label
    fp.close()
    
    print("Initial Trainingset size :  ", len(trainsetl))
    print("Initial Validationset size: ", len(validsetl))
    print("Initial Testset size      : ", len(testsetl))
    
    Xtrainl = []
    ytrainl = []
    Trainmaptoid = {}
    train_molnames = []
    
    Xtestl = []
    ytestl = []
    Testmaptoid = {}
    test_molnames = []
    
    Xvalidl = []
    yvalidl = []
    Validmaptoid = {}
    valid_molnames = []
    
    Xfulll = []
    yfulll = []
    
    conformerstolabel = {}
    
    for index, row in df.iterrows():
        name = row.values[0]
        basename = re.split(npzsplit, name)[0]
    
        print(name, basename)
    
        conformerstolabel[name] = moltolabels[basename]
    
        if basename in trainsetl:
            x = np.copy(row.values[1:])
            x = x.astype(float)
            #x = x[np.logical_not(np.isnan(x))]
            x = np.nan_to_num (x)
            Xtrainl.append(x.tolist())
    
            yval = moltolabels[basename]
            ytrainl.append(yval)
            train_molnames.append(name)
            if not (basename in Trainmaptoid):
                Trainmaptoid[basename] = []
            Trainmaptoid[basename].append(len(Xtrainl)-1)
    
            Xfulll.append(x.tolist())
            yfulll.append(yval)
    
            #print("Molecule in training", row.values[0], " logBB: ", yval)
        elif basename in testsetl:
            x = np.copy(row.values[1:])
            x = x.astype(float)
            #x = x[np.logical_not(np.isnan(x))]
            x = np.nan_to_num (x)
            Xtestl.append(x.tolist())
    
            yval = moltolabels[basename]
            ytestl.append(yval)
            test_molnames.append(name)
            if not (basename in Testmaptoid):
                Testmaptoid[basename] = []
            Testmaptoid[basename].append(len(Xtestl)-1)
    
            Xfulll.append(x.tolist())
            yfulll.append(yval)
    
            #print("Molecule in test", row.values[0], " logBB: ", yval)
        elif basename in validsetl:
            x = np.copy(row.values[1:])
            x = x.astype(float)
            #x = x[np.logical_not(np.isnan(x))]
            x = np.nan_to_num (x)
            Xvalidl.append(x.tolist())
    
            yval = moltolabels[basename]
            yvalidl.append(yval)
            valid_molnames.append(name)
            if not (basename in Validmaptoid):
                Validmaptoid[basename] = []
            Validmaptoid[basename].append(len(Xvalidl)-1)
    
            Xfulll.append(x.tolist())
            yfulll.append(yval)
    
            #print("Molecule in valid", row.values[0], " logBB: ", yval)
        else:
            print("Molecule ", name, " ", basename, " not found")
    
    
    #for mol in Trainmaptoid:
    #    print(mol)
    #    for idx in Trainmaptoid[mol]:
    #        print("   ", idx)
    
    X_train = np.asanyarray(Xtrainl)
    y_train = np.asanyarray(ytrainl)
    y_train = y_train.astype(float)
    
    X_test = np.asanyarray(Xtestl)
    y_test = np.asanyarray(ytestl)
    y_test = y_test.astype(float)
    
    X_valid = np.asanyarray(Xvalidl)
    y_valid = np.asanyarray(yvalidl)
    y_valid = y_valid.astype(float)
    
    X_full = np.asanyarray(Xfulll)
    y_full = np.asanyarray(yfulll)
    y_full = y_full.astype(float)
    
    print("Training   ", X_train.shape, y_train.shape)
    print("Validation ", X_valid.shape, y_valid.shape)
    print("Test       ", X_test.shape, y_test.shape)
    
    n_estimators = [50, 100, 300, 500, 800, 1200]
    max_depth = [None, 5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [10, 20, 50, 100, 200] 
    random_state = [42]
    max_features = ['auto', 'sqrt']
    bootstrap = [True]
    
    # quick test
    if QUICKTEST:
        n_estimators = [500]
        max_depth = [5]
        min_samples_split = [2]
        min_samples_leaf = [50] 
        random_state = [1]
        max_features = ['sqrt']
        bootstrap = [True]
    
    hyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth, 
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}
    
    besthyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}
    
    foudoneatleat = False
    total = 1
    for k in hyperF:
        total *= len(hyperF[k])
    counter = 1
    bestmse = float("+inf")
    bestmse_test = float("+inf")
    bestmse_diff =  float("+inf")
    best_valid_gmfe_logBB =  float("+inf")
                                
    print ("ID , Train MSE , Validation NSE, Diff, Perc Diff, GMFEValid, GMFETrain, n_estimators, " + \
        "max_depth , min_samples_split , min_samples_leaf , max_features ")
    for a in hyperF["n_estimators"]:
        for b in  hyperF["max_depth"]:
            for c in  hyperF["min_samples_split"]:
                for d in  hyperF["min_samples_leaf"]:
                    for e in  hyperF["random_state"]:
                        for f in  hyperF["bootstrap"]:
                            for g in  hyperF["max_features"]:
                                model = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                )
                                model.fit(X_train, y_train)
    
                                y_pred = model.predict(X_train)
                                train_mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
    
                                train_gmfe_logBB, pc_good2, pc_good3, afe_logBB = \
                                    pcsv.gmfe_logBB(y_train, y_pred)
    
                                y_pred = model.predict(X_valid)
                                valid_mse = sklearn.metrics.mean_squared_error(y_valid, y_pred)
    
                                valid_gmfe_logBB, pc_good2, pc_good3, afe_logBB = \
                                    pcsv.gmfe_logBB(y_valid, y_pred)
                                
                                diffmse = math.fabs(train_mse - valid_mse)
                                percdiff = 100.0*(diffmse/((train_mse+valid_mse)/2))
    
                                print("%10d of %10d , %10.5f , %10.5f , %10.5f , %10.5f , %10.5f , %10.5f ,"%(\
                                    counter , total, train_mse, valid_mse, diffmse, percdiff , \
                                    valid_gmfe_logBB, train_gmfe_logBB), end="")
                                print(a, " , " , b , " , " , c , " , " , d , " , " , g)
    
                                sys.stdout.flush()
    
                                counter += 1
                                if optimizegmfe:
                                    if (valid_gmfe_logBB < best_valid_gmfe_logBB):
    
                                        bestmse = train_mse
                                        bestmse_test = valid_mse
                                        bestmse_diff = diffmse
    
                                        best_valid_gmfe_logBB = valid_gmfe_logBB
                                        
                                        besthyperF = {"n_estimators" : a,
                                                     "max_depth" : b,  
                                                     "min_samples_split" : c, 
                                                     "min_samples_leaf" : d, 
                                                     "random_state" : e, 
                                                     "bootstrap" : f,
                                                     "max_features" : g}
                                        
                                        foudoneatleat = True
                                else:
                                    if percdiff <= 120.0:
                                    #if train_mse < bestmse and test_mse < bestmse_test and \
                                    #    diffmse < bestmse_diff:
                                        if valid_mse < bestmse_test:
                                    
                                            bestmse = train_mse
                                            bestmse_test = valid_mse
                                            bestmse_diff = diffmse
                                        
                                            besthyperF = {"n_estimators" : a,
                                                     "max_depth" : b,  
                                                     "min_samples_split" : c, 
                                                     "min_samples_leaf" : d, 
                                                     "random_state" : e, 
                                                     "bootstrap" : f,
                                                     "max_features" : g}
                                        
                                            foudoneatleat = True
    
    
    if not foudoneatleat:
        print("Grid search failed")
    
        if QUICKTEST:
            besthyperF = {"n_estimators" : n_estimators[0], 
             "max_depth" : max_depth[0],  
             "min_samples_split" : min_samples_split[0], 
             "min_samples_leaf" : min_samples_leaf[0], 
             "random_state" : random_state[0], 
             "bootstrap" : bootstrap[0],
             "max_features" : max_features[0]}
        else:
            exit(0)
                             
    print ("Best  MSE Train: ", bestmse)
    print ("Best  MSE Valid: ", bestmse_test)
    print ("Best GMFE Valid: ", best_valid_gmfe_logBB)
    
    model = RandomForestRegressor(**besthyperF)
    
    print(model.get_params())
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    y_pred_train = model.predict(X_train)
    mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
    rmse = math.sqrt(mse)
    print("Training Set size: ", len(y_train), " ", len(y_pred))
    
    fp = open("RF_trainset_precictions.csv", "w")
    
    fp.write("Molname ; Predicted ; Truevalue\n")
    for i, pv in enumerate(y_pred):
        name = train_molnames[i]
        fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, conformerstolabel[name]))
    
    fp.close()
    
    best_train_predictions, avg_train_predictions, \
        std_train_predictions, train_singlevals , best_mse, avg_mse, moltodiff = \
            extract_predicions (Trainmaptoid, y_train, y_pred)
    
    plot_scatter ( 'Training Set, Best Predictions', 'Experimental', 'Predicted', \
        train_singlevals, best_train_predictions, 'b.', 0.4, "RF_train_scatter_best.png" )
    
    plot_scatter_with_errors ( 'Training Set, Average Predictions', 'Experimental', \
        'Predicted', train_singlevals, avg_train_predictions, std_train_predictions, \
            'b.', 'b', 0.4, "RF_train_scatter_avg.png" )
    
    plot_scatter ( 'Training Set, All Predictions', 'Experimental', 'Predicted', \
        y_train, y_pred, 'b.', 0.3, "RF_train_scatter.png" )
    
    print("RF Traininig      MSE: ", mse)
    print("RF Traininig Best MSE: ", best_mse)
    print("RF Traininig  Avg MSE: ", avg_mse)
    
    y_pred = model.predict(X_test)
    y_pred_test = model.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print("Test Set size: ", len(y_test), " ", len(y_pred))
    
    fp = open("RF_testset_precictions.csv", "w")
    
    fp.write("Molname , Predicted , Truevalue\n")
    for i, pv in enumerate(y_pred):
        name = test_molnames[i]
        fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, conformerstolabel[name]))
    
    fp.close()
    
    best_test_predictions, avg_test_predictions, \
        std_test_predictions, test_singlevals , best_mse, avg_mse, moltodiff = \
            extract_predicions (Testmaptoid, y_test, y_pred)
    
    plot_scatter ( 'Test Set, Best Prediction', 'Experimental', 'Predicted', \
        test_singlevals, best_test_predictions, 'r.', 0.4, "RF_test_scatter_best.png" )
    
    plot_scatter_with_errors ( 'Test Set, Average Prediction', 'Experimental', 'Predicted', \
        test_singlevals, avg_test_predictions, std_test_predictions, 'r.', 'r', 0.4, "RF_test_scatter_avg.png" )
    
    plot_scatter ( 'Test Set, All Predictions', 'Experimental', 'Predicted', \
        y_test, y_pred, 'r.', 0.3, "RF_test_scatter.png" )
    
    print("RF Test      MSE: ", mse)
    print("RF Test Best MSE: ", best_mse)
    print("RF Test  Avg MSE: ", avg_mse)
    
    y_pred = model.predict(X_valid)
    y_pred_valid = model.predict(X_valid)
    mse = sklearn.metrics.mean_squared_error(y_valid, y_pred)
    rmse = math.sqrt(mse)
    print("Validation Set size: ", len(y_valid), " ", len(y_pred))
    
    fp = open("RF_validset_precictions.csv", "w")
    
    fp.write("Molname , Predicted , Truevalue\n")
    for i, pv in enumerate(y_pred):
        name = valid_molnames[i]
        fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, conformerstolabel[name]))
    
    fp.close()
    
    best_valid_predictions, avg_valid_predictions, \
        std_valid_predictions, valid_singlevals , best_mse, avg_mse, moltodiff = \
            extract_predicions (Validmaptoid, y_valid, y_pred)
    
    
    plot_scatter ( 'Validation Set, Best Predictions', 'Experimental', 'Predicted', \
        valid_singlevals, best_valid_predictions, 'g.', 0.4, "RF_valid_scatter_best.png" )
    
    plot_scatter_with_errors ( 'Validation Set, Average Predictions', 'Experimental', \
        'Predicted', valid_singlevals, avg_valid_predictions, std_valid_predictions, \
            'g.', 'g', 0.4, "RF_valid_scatter_avg.png" )
    
    plot_scatter ( 'Validation Set, All Predictions', 'Experimental', 'Predicted', y_valid,\
         y_pred, 'g.', 0.3, "RF_valid_scatter.png" )
    
    print("RF Validation      MSE: ", mse)
    print("RF Validation Best MSE: ", best_mse)
    print("RF Validation  Avg MSE: ", avg_mse)
    
    featimp = {}
    if QUICKTEST:
        results = permutation_importance(model, X_train, y_train, n_repeats=1, random_state=0,
                scoring="neg_mean_squared_error", n_jobs=1)
    
        importance = results.importances_mean
        importanceerror = results.importances_std
    
        for i, c in enumerate(featuresusedlist):
            print(i, c)
            featimp[c] = importance[i]
    else:
        results = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=0,
                scoring="neg_mean_squared_error", n_jobs=4)
    
        importance = results.importances_mean
        importanceerror = results.importances_std
    
        for i, c in enumerate(featuresusedlist):
            print(i, c)
            featimp[c] = importance[i]
    
    sorted_featimp = dict(sorted(featimp.items(), key=lambda item: item[1]))
    
    for c in sorted_featimp:
        print("Feat %10s %10.5f "%(c, sorted_featimp[c]))
    
    #print(importance.shape)
    #print(importanceerror.shape)
    #print(len(featuresusedlist))
    
    # plot feature importance
    pyplot.clf()
    pyplot.figure(figsize=(10, 10))
    pyplot.rcParams.update({'font.size': 8})
    pyplot.title(
        "Features importance from Permutation [neg_mean_squared_error]")
    pyplot.barh(featuresusedlist, importance, xerr=importanceerror, capsize=10)
    pyplot.xticks(rotation=45, ha="right")
    pyplot.gcf().subplots_adjust(bottom=0.30)
    pyplot.savefig("RF_permutation_importance.png", dpi=600)
    pyplot.close()
     
    #r2s = []
    mses = []
    gmfes = []
    ncomps = []
    
    for ncomp in range(1,20):
        pls = PLSRegression(ncomp)
        pls.fit(X_train, y_train)
    
        y_pred = pls.predict(X_valid)
    
        mse = sklearn.metrics.mean_squared_error(y_valid, y_pred)
        rmse = math.sqrt(mse)
    
        valid_gmfe_logBB, pc_good2, pc_good3, afe_logBB = \
                                    pcsv.gmfe_logBB(y_valid, y_pred)
    
        # in questo modo valuta usando leave one out 
        #cv = LeaveOneOut()
    
        #by splitting the data, fitting a model and computing the score 10 consecutive times (with different splits each time):
        #cv = 10
    
        #scores = cross_val_score(pls, X_full, y_full, scoring='neg_mean_squared_error',
        #                     cv=cv, n_jobs=-1)
        
        #y_cv = cross_val_predict(pls, X_train, y_train, cv=10)
        #score = r2_score(y_train, y_cv)
        #mse = mean_squared_error(y_train, y_cv)
    
        #rmse = np.sqrt(np.mean(np.absolute(scores)))
    
        #scores = cross_val_score(pls, X_full, y_full, scoring='r2',
        #                     cv=cv, n_jobs=-1)
    
        # posso usare cross_val_predict con  LeaveOneOut per poi calcolare r2 e quindi q2 direi
        # vedi qui: https://scikit-learn.org/stable/modules/cross_validation.html 
        
        #print(r2score)
        #r2score = np.mean(scores)
        #print(len(scores))
    
        #r2s.append(r2score)
        
        mses.append(mse)
        ncomps.append(ncomp)
        gmfes.append(valid_gmfe_logBB)
        print("%4d %10.8f  %10.8f"%(ncomp, mse, valid_gmfe_logBB))
        sys.stdout.flush()
    
    
    msemin = None 
    
    if optimizegmfe:
        msemin = np.argmin(gmfes)
        print("Suggested number of components: ", msemin+1)
    else:
        msemin = np.argmin(mses)
        print("Suggested number of components: ", msemin+1)
    
    pyplot.clf()
    pyplot.rcParams.update({'font.size': 15})
    #pyplot.plot(ncomps, r2s, '-o', color='black')
    pyplot.plot(ncomps, mses, '-o', color='black')
    pyplot.xlabel('Number of Components')
    pyplot.ylabel('MSE')
    pyplot.xticks(ncomps)
    pyplot.savefig("PLS_components_MSE.png", bbox_inches="tight", dpi=600)
    
    ncomp = msemin+1
    pls = PLSRegression(ncomp)
    pls.fit(X_train, y_train)
    
    y_pred_train = pls.predict(X_train)
    
    fp = open("PLS_trainset_precictions.csv", "w")
    
    fp.write("Molname , Predicted , Truevalue\n")
    for i, pv in enumerate(y_pred_train):
        name = train_molnames[i]
        fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, conformerstolabel[name]))
    
    fp.close()
    
    mse = mean_squared_error(y_train, y_pred_train)
    
    best_train_predictions, avg_train_predictions, \
        std_train_predictions, train_singlevals , best_mse, avg_mse, moltodiff = \
            extract_predicions (Trainmaptoid, y_train, y_pred_train)
    
    plot_scatter ( 'Training Set, Best Predictions', 'Experimental', \
        'Predicted', train_singlevals, best_train_predictions, 'b.', 0.4, \
            "PLS_train_scatter_best.png" )
    
    plot_scatter_with_errors ( 'Training Set, Average Predictions', 'Experimental', \
        'Predicted', train_singlevals, avg_train_predictions, std_train_predictions, \
            'b.', 'b', 0.4, "PLS_train_scatter_avg.png" )
    
    plot_scatter ( 'Training Set, All Predictions', 'Experimental', 'Predicted', \
        y_train, y_pred_train, 'b.', 0.3, "PLS_train_scatter.png" )
    
    print("PLS Traininig      MSE: ", mse)
    print("PLS Traininig Best MSE: ", best_mse)
    print("PLS Traininig  Avg MSE: ", avg_mse)
    
    y_pred_test = pls.predict(X_test)
    
    fp = open("PLS_testset_precictions.csv", "w")
    
    fp.write("Molname , Predicted , Truevalue\n")
    for i, pv in enumerate(y_pred_test):
        name = test_molnames[i]
        fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, conformerstolabel[name]))
    
    fp.close()
    
    mse = mean_squared_error(y_test, y_pred_test)
    
    best_test_predictions, avg_test_predictions, \
        std_test_predictions, test_singlevals , best_mse, avg_mse, moltodiff = \
            extract_predicions (Testmaptoid, y_test, y_pred_test)
    
    plot_scatter ( 'Test Set, Best Predictions', 'Experimental', 'Predicted', \
        test_singlevals, best_test_predictions, 'r.', 0.4, "PLS_test_scatter_best.png" )
    
    plot_scatter_with_errors ( 'Test Set, Average Predictions', 'Experimental', \
        'Predicted', test_singlevals, avg_test_predictions, std_test_predictions,\
             'r.', 'r', 0.4, "PLS_test_scatter_avg.png" )
    
    plot_scatter ( 'Test Set, All Predictions', 'Experimental', 'Predicted', \
        y_test, y_pred_test, 'r.', 0.3, "PLS_test_scatter.png" )
    
    print("PLS Test      MSE: ", mse)
    print("PLS Test Best MSE: ", best_mse)
    print("PLS Test  Avg MSE: ", avg_mse)
     
    y_pred_valid = pls.predict(X_valid)
    
    fp = open("PLS_validset_precictions.csv", "w")
    
    fp.write("Molname , Predicted , Truevalue\n")
    for i, pv in enumerate(y_pred_valid):
        name = valid_molnames[i]
        fp.write("%s ; %10.5f ; %10.5f\n"%(name, pv, conformerstolabel[name]))
    
    fp.close()
    
    mse = mean_squared_error(y_valid, y_pred_valid)
    
    best_valid_predictions, avg_valid_predictions, \
        std_valid_predictions, valid_singlevals , best_mse, avg_mse, moltodiff = \
            extract_predicions (Validmaptoid, y_valid, y_pred_valid)
    
    plot_scatter ( 'Validation Set, Best Predictions', 'Experimental', \
        'Predicted', valid_singlevals, best_valid_predictions, 'g.', 0.4, \
            "PLS_valid_scatter_best.png" )
    
    plot_scatter_with_errors ( 'Validation Set, Average Predictions', 'Experimental', \
        'Predicted', valid_singlevals, avg_valid_predictions, std_valid_predictions,\
             'g.', 'g', 0.4, "PLS_valid_scatter_avg.png" )
    
    plot_scatter ( 'Validation Set, All Predictions', 'Experimental', 'Predicted', \
        y_valid, y_pred_valid, 'g.', 0.3, "PLS_valid_scatter.png" )
    
    print("PLS Validation      MSE: ", mse)
    print("PLS Validation Best MSE: ", best_mse)
    print("PLS Validation  Avg MSE: ", avg_mse)
