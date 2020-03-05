#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:45:08 2020

@author: inpace
"""
import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import numpy as np

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1),
                     padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  matrics=['acc'])
    return model

x = np.load('X_dataset.npy')
x = np.resize(-1, (531,13,112))
y = np.load('Y_dataset.npy')

y_flat = np.argmax(y, axis=1)

input_shape = (x.shape[1], x.shape[2], 1)

class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

model = get_conv_model()

model.fit(x, y, epochs=10, batch_size=32, shuffle=True)