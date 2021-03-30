# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_nb.py  朴素贝叶斯
"""
import numpy as np
import matplotlib.pyplot as mp

# 读取样本数据
data = np.loadtxt(
    '../ml_data/multiple1.txt', delimiter=',')
x = data[:, :-1]
y = data[:,  -1]
print(x.shape, y.shape)

# 训练朴素贝叶斯模型
import sklearn.naive_bayes as nb
model = nb.GaussianNB()
model.fit(x, y)
# 针对训练集样本进行预测
pred_y = model.predict(x)
print('acc:', (y == pred_y).sum() / y.size)

# 画图：
mp.figure('NB Classification', facecolor='lightgray')
mp.title('NB Classification', fontsize=16)

# 绘制分类边界线
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), np.linspace(b, t, n))
# 预测每个点的类别标签
grid_z = model.predict(np.column_stack(
    (grid_x.ravel(), grid_y.ravel())))
grid_z = grid_z.reshape(grid_x.shape)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')

mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg_r',
           label='Samples', s=80)
mp.legend()
mp.show()
