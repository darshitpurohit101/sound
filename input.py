# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:30:52 2020

@author: Darshit.Purohit
"""

import sounddevice as sd
from scipy.io.wavfile import write
import os

def record():
    fs = 25000
    second = 10
    record = sd.rec(int(second*fs), 
                    samplerate = fs,
                    channels=2)
    print("Recoder Started")
    sd.wait()
    print("Fnished")
    write('output' + '.wav', fs, record)
        
record()