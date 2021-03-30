# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_load.py 模型加载
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import pickle
# 造数据
test_x = np.linspace(1, 8, 20).reshape(-1, 1)
print(test_x.shape)

with open('linear.model', 'rb') as f:
    model = pickle.load(f)

# 使用模型预测结果
pred_test_y = model.predict(test_x)
print(np.column_stack((test_x, pred_test_y)))
