# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_polyfit.py 多项式回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

# 读取数据
train_x, train_y = np.loadtxt(
    '../ml_data/single.txt', delimiter=',',
    usecols=(0, 1), unpack=True)
train_x = train_x.reshape(-1, 1)
print(train_x.shape, train_y.shape)

# 训练多项式回归模型
import sklearn.pipeline as pl
import sklearn.preprocessing as sp

model = pl.make_pipeline(
    sp.PolynomialFeatures(8),
    lm.LinearRegression())

model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)

# 评估模型误差
import sklearn.metrics as sm
print(sm.mean_absolute_error(train_y, pred_train_y))
print(sm.mean_squared_error(train_y, pred_train_y))
print(sm.median_absolute_error(train_y, pred_train_y))
print('r2:', sm.r2_score(train_y, pred_train_y))

# 绘图
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, color='dodgerblue',
           s=80, label='Samples')

# 绘制回归线
px = np.linspace(train_x.min(), train_x.max(), 1000)
py = model.predict(px.reshape(-1, 1))
mp.plot(px, py, color='orangered',
        label='Regression Line')
mp.legend()
mp.show()
