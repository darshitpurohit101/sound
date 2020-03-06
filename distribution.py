#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 23:49:48 2020

@author: inpace
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
import numpy as np

df = pd.read_csv('label.csv')
df.set_index('File name', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('/home/inpace/Desktop/IBM project/clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.Label))
class_dist = df.groupby(['Label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()