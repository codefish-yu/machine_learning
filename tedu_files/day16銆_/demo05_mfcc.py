# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_mfcc.py  梅尔频率倒谱系数
"""
import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import matplotlib.pyplot as mp

# 读取音频文件
sample_rate, sigs = wf.read(
    '../ml_data/speeches/training/banana/banana01.wav')
print(sample_rate, sigs.shape)
mfcc = sf.mfcc(sigs, sample_rate)
print(mfcc.shape)

mp.imshow(mfcc.T, cmap='gist_rainbow')
mp.show()
