# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_cls.py  分类业务
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.array([
    [3, 1],
    [2, 5],
    [1, 8],
    [6, 4],
    [5, 2],
    [3, 5],
    [4, 7],
    [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

# 画图：
mp.figure('Classification', facecolor='lightgray')
mp.title('Classification', fontsize=16)

# 绘制分类边界线
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), np.linspace(b, t, n))
# 预测每个点的类别标签
grid_z = np.zeros_like(grid_x)
# 若x坐标大于y坐标，则划分到1类别
grid_z[grid_x < grid_y] = 1
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')

mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg',
           label='Samples', s=80)

mp.legend()
mp.show()
