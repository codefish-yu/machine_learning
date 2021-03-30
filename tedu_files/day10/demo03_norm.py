# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_norm.py   归一化
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])

r = sp.normalize(raw_samples, norm='l1')
print(r)
# 水平方向求和
print(np.sum(r, axis=1))

r = sp.normalize(raw_samples, norm='l2')
print(r)
# 水平方向求和
print(np.sum(r**2, axis=1))
