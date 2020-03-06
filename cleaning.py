
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:06:51 2020

@author: redcape
"""

import librosa
import noisereduce as nr
#import matplotlib.pyplot as plt
#import numpy as np
import glob
#import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile
import file_label_csv

clean_list = ['dog_bark', 'gun_shot', 'siren', 'car_horn', 'street_music']

for name in clean_list:
    file_name = glob.glob("/home/inpace/Desktop/IBM project/UrbanSound/data/"+name+"/*.wav")
    counter = 0
    
    
    for i in tqdm(file_name):
        print(i)
        #Load audio file
        ''' can also be done as
        data, rate = wavfile.read('filename')'''
        data, rate = librosa.load(i)
        
        counter += 1
        #Noice reduction
        noise_part = data[0:25000]
        noise_reduce = nr.reduce_noise(audio_clip=data, noise_clip=noise_part, verbose=False)
        
        #triming the scilence part from the audio
        trimmed, index = librosa.effects.trim(noise_reduce, top_db=20, frame_length=512, hop_length=64)
        
        save_file_name = name+"{}.wav".format(counter)
        file_label_csv.datawrite(save_file_name, name)
        wavfile.write(filename = '/home/inpace/Desktop/IBM project/clean/'+save_file_name, rate=rate, data = trimmed)
        
        
