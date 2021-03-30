# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_lr.py 线性回归
"""
import numpy as np
import matplotlib.pyplot as mp

xs = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
ys = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

times = 1000
w0, w1 = 1, 1
lrate = 0.01
for i in range(1, times + 1):
    # 计算w0方向上的偏导数
    d0 = (w0 + w1 * xs - ys).sum()
    # 计算w1方向上的偏导数
    d1 = (xs * (w0 + w1 * xs - ys)).sum()
    # 让w0与w1随着for循环不断更新
    w0 = w0 - lrate * d0
    w1 = w1 - lrate * d1

# 画图显示效果：
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(xs, ys, s=80, color='dodgerblue',
           label='Samples')
# 通过w0与w1 绘制回归线
pred_y = w0 + w1 * xs
mp.plot(xs, pred_y, color='orangered',
        label='Regression Line')
mp.legend()
mp.show()
