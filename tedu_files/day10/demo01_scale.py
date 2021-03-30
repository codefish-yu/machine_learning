# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_scale.py
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])

# 均值移除
r = sp.scale(raw_samples)
print(r)
# 每一列均值为0，标准差为1
print(r.mean(axis=0))
print(r.std(axis=0))
