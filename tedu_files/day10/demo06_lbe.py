# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_lbe.py  标签编码
"""
import numpy as np
import sklearn.preprocessing as sp

samples = ['Audi', 'Bmw', 'Benz', 'Bmw', 'Audi',
           'Toyota', 'Ford', 'Toyota', 'BYD', 'Ford',
           'Audi']
# 标签编码器
encoder = sp.LabelEncoder()
r = encoder.fit_transform(samples)
print(r)

# 假设通过模型预测后得到一组预测结果：
pred = [0, 1, 2, 3, 4, 3, 1, 2, 3, 4]
pred_text = encoder.inverse_transform(pred)
print(pred_text)
