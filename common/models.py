import os
import math
import numpy as np

import tensorflow as tf
#tf.config.experimental.set_lms_enabled(True)

#import keras

from tensorflow import keras 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, LeakyReLU

from tensorflow.keras import regularizers

#import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#from keras import backend as K
from tensorflow.keras import backend as K 


######################################################################################

def model_scirep_selection_hyperopt(iinput_shape, ndense_layers, nunits, lnfilters,
    ksize, psize, threeconv):
    """
    Architecture from
    https://doi.org/10.1038/s41598-017-17299-w
    """

    l2val = 0.001

    model = Sequential()
    model.add(Conv3D(lnfilters[0],
                         kernel_size=ksize,
                         input_shape=iinput_shape,
                         kernel_regularizer=regularizers.l2(l=l2val), 
                         padding='same'))
  
    model.add(LeakyReLU()) # schiaaiano i valori negativi forse nel nostro caso non ne vale la pena 
    #model.add(BatchNormalization()) # anche qui normalizzazione, serve ?
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=psize))

    if threeconv:

        model.add(Conv3D(lnfilters[1], kernel_size=ksize,
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.2, input_shape=iinput_shape))
        model.add(MaxPooling3D(pool_size=psize))

    model.add(Conv3D(lnfilters[2], kernel_size=ksize,
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation="sigmoid",
                kernel_regularizer=regularizers.l2(l=l2val))) # activation="sigmoid" for classification
        model.add(LeakyReLU()) # qui serve perche; abbiamo una funzione di attivazione lineare 
        model.add(Dropout(0.2, input_shape=(nunits,)))
    
    model.add(Dense(2, activation="sigmoid")) # activation="sigmoid" for classification
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',\
         metrics=['accuracy'])
    
    return model

######################################################################################

def model_scirep_selection(iinput_shape, ndense_layers, nunits, nfilters):
    """
    Architecture from
    https://doi.org/10.1038/s41598-017-17299-w
    """

    l2val = 0.001

    model = Sequential()
    model.add(Conv3D(nfilters,
                         kernel_size=(3, 3, 3),
                         input_shape=iinput_shape,
                         kernel_regularizer=regularizers.l2(l=l2val),
                         padding='same' ))
  
    model.add(LeakyReLU()) # schiaaiano i valori negativi forse nel nostro caso non ne vale la pena 
    #model.add(BatchNormalization()) # anche qui normalizzazione, serve ?
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation="sigmoid",
                kernel_regularizer=regularizers.l2(l=l2val))) # activation="sigmoid" for classification
        model.add(LeakyReLU()) # qui serve perche; abbiamo una funzione di attivazione lineare 
        model.add(Dropout(0.2, input_shape=(nunits,)))
    
    # regression in case

    model.add(Dense(2, activation="sigmoid")) # activation="sigmoid" for classification
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', \
        metrics=['accuracy'])

    return model

######################################################################################

def model_scirep(input_shape, ndense_layers, nunits, nfilters):
    """
    Architecture from
    https://doi.org/10.1038/s41598-017-17299-w
    """
    model = Sequential()
    model.add(Conv3D(nfilters,
                         kernel_size=(3, 3, 3),
                         input_shape=input_shape))
  
    #model.add(LeakyReLU()) # schiaaiano i valori negativi forse nel nostro caso non ne vale la pena 
    #model.add(BatchNormalization()) # anche qui normalizzazione, serve ?
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     input_shape=input_shape))
    #model.add(LeakyReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     input_shape=input_shape))
    #model.add(LeakyReLU())

    model.add(Flatten())
    
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation="sigmoid")) # activation="sigmoid" for classification
        #model.add(LeakyReLU()) # qui serve perche; abbiamo una funzione di attivazione lineare 
    
    # regression in case
    #model.add(Dense(1))
    #model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'], 
    #             metrics=['mse', 'mae'])
    model.add(Dense(2, activation="sigmoid")) # activation="sigmoid" for classification
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

######################################################################################

def model_scirep_regression(iinput_shape, ndense_layers, nunits, nfilters):
    """
    Architecture from
    https://doi.org/10.1038/s41598-017-17299-w
    """

    l2val = 0.001

    model = Sequential()
    model.add(Conv3D(nfilters,
                         kernel_size=(3, 3, 3),
                         input_shape=iinput_shape,
                         kernel_regularizer=regularizers.l2(l=l2val),
                         padding='same' ))
  
    model.add(LeakyReLU()) # schiaaiano i valori negativi forse nel nostro caso non ne vale la pena 
    #model.add(BatchNormalization()) # anche qui normalizzazione, serve ?
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(nfilters, kernel_size=(3, 3, 3),
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation="linear",
                kernel_regularizer=regularizers.l2(l=l2val))) # activation="sigmoid" for classification
        model.add(LeakyReLU()) # qui serve perche; abbiamo una funzione di attivazione lineare 
        model.add(Dropout(0.2, input_shape=(nunits,)))
    
    # regression in case
    model.add(Dense(1))
            #kernel_regularizer=regularizers.l2(l=0.01))) # activation="sigmoid" for classification
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', 
                 metrics=['mse', 'mae'])
 
    
    return model

######################################################################################

def model_scirep_regression_hyperopt(iinput_shape, ndense_layers, nunits, lnfilters,
    ksize, psize, threeconv):
    """
    Architecture from
    https://doi.org/10.1038/s41598-017-17299-w
    """

    l2val = 0.001

    model = Sequential()
    model.add(Conv3D(lnfilters[0],
                         kernel_size=ksize,
                         input_shape=iinput_shape,
                         kernel_regularizer=regularizers.l2(l=l2val), 
                         padding='same'))
  
    model.add(LeakyReLU()) # schiaaiano i valori negativi forse nel nostro caso non ne vale la pena 
    #model.add(BatchNormalization()) # anche qui normalizzazione, serve ?
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size=psize))

    if threeconv:

        model.add(Conv3D(lnfilters[1], kernel_size=ksize,
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.2, input_shape=iinput_shape))
        model.add(MaxPooling3D(pool_size=psize))

    model.add(Conv3D(lnfilters[2], kernel_size=ksize,
                     kernel_regularizer=regularizers.l2(l=l2val),
                     padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation="linear",
                kernel_regularizer=regularizers.l2(l=l2val))) # activation="sigmoid" for classification
        model.add(LeakyReLU()) # qui serve perche; abbiamo una funzione di attivazione lineare 
        model.add(Dropout(0.2, input_shape=(nunits,)))
    
    # regression in case
    model.add(Dense(1))
            #kernel_regularizer=regularizers.l2(l=0.01))) # activation="sigmoid" for classification
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', 
                 metrics=['mse', 'mae'])
 
    
    return model

######################################################################################
