import os
import glob
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from python_speech_features import mfcc
from keras.utils import to_categorical

df = pd.read_csv('label.csv') 
sample = df['length']
classes = list(np.unique(df.Label))
class_dist = df.groupby(['Label'])['length'].mean()
prob_dist = class_dist/class_dist.sum()

nfeat = 13
nfilt = 26
nfft = 615

x = []
y = []
 
_min, _max = float('inf'), -float('inf')
for _ in tqdm(range(sample)):
    
    rand_class = np.random.choice(class_dist.index, p = prob_dist)
    file = np.random.choice(df[df.Label==rand_class].index)
    rate, wav= wavfile.read('/home/inpace/Desktop/IBM project/clean/'+file)
    label = df.at[file, 'Label' ]
    sample = wav[0:25000]
    X_sample = mfcc(sample, rate, numcep=nfeat, nfilt=nfilt, nfft=nfft ).T
    _min = min(np.amin(X_sample), _min)
    _max = max(np.amax(X_sample), _max)
    x.append(X_sample)
    y.append(classes.index(label))

x, y = np.array(x), np.array(y)
x = (x - _min)/(_max - _min)
x = x.reshape(x.reshape[0], x.reshape[1]. x.reshape[2], 1)
y = to_categorical(y, num_classes=5)

np.save('X_dataset.npy', x)
np.save('Y_dataset.npy', y)    


