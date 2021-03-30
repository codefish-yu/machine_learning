# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_save.py 模型保存
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import pickle

# 读取数据
train_x, train_y = np.loadtxt(
    '../ml_data/single.txt', delimiter=',',
    usecols=(0, 1), unpack=True)
train_x = train_x.reshape(-1, 1)
# 训练模型
model = lm.LinearRegression()
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)

# 评估模型误差
import sklearn.metrics as sm
print(sm.mean_absolute_error(train_y, pred_train_y))
print(sm.mean_squared_error(train_y, pred_train_y))
print(sm.median_absolute_error(train_y, pred_train_y))
print('r2:', sm.r2_score(train_y, pred_train_y))

# 保存模型
with open('linear.model', 'wb') as f:
    pickle.dump(model, f)

print('save "linear.model" success.')
