# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_mms.py   MinMaxScaler 范围缩放
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])

# 自己实现范围缩放
# 遍历每一列
cols = raw_samples.shape[1]
r = []
for i in range(cols):
    col_val = raw_samples[:, i]  # 切出一列
    # 求得适合这一列的k与b，计算缩放结果
    A = np.array([[col_val.min(), 1],
                  [col_val.max(), 1]])
    B = np.array([[0], [1]])
    x = np.linalg.lstsq(A, B)[0]
    r.append(col_val * x[0] + x[1])
print(np.array(r).T)


mms = sp.MinMaxScaler(feature_range=(0, 1))
r = mms.fit_transform(raw_samples)
print(r)
