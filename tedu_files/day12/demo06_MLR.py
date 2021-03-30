# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_MLR.py  逻辑回归实现多元分类
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([
    [4, 7],
    [3.5, 8],
    [3.1, 6.2],
    [0.5, 1],
    [1, 2],
    [1.2, 1.9],
    [6, 2],
    [5.7, 1.5],
    [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# 训练LR分类模型
import sklearn.linear_model as lm
model = lm.LogisticRegression(
    solver='liblinear', C=100)
model.fit(x, y)

# 画图：
mp.figure('Classification', facecolor='lightgray')
mp.title('Classification', fontsize=16)

# 绘制分类边界线
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), np.linspace(b, t, n))
print(grid_x.shape, grid_y.shape)
# 预测每个点的类别标签
grid_z = model.predict(np.column_stack(
    (grid_x.ravel(), grid_y.ravel())))
grid_z = grid_z.reshape(grid_x.shape)

mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg_r',
           label='Samples', s=80)
mp.legend()
mp.show()
