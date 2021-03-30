# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_svm_linear.py  基于线性核函数的SVM
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.model_selection as ms
import sklearn.svm as svm

# 读取样本数据
data = np.loadtxt(
    '../ml_data/multiple2.txt', delimiter=',')
x = data[:, :-1]
y = data[:,  -1]
print(x.shape, y.shape)

# 拆分训练集与测试集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(
        x, y, test_size=0.25, random_state=7)

# 训练朴素贝叶斯模型
model = svm.SVC(kernel='rbf', gamma=0.01, C=600)
model.fit(train_x, train_y)

# 针对测试集样本进行预测
pred_test_y = model.predict(test_x)
print('acc:', (test_y == pred_test_y).sum() / test_y.size)
# 输出错误样本的特征数据
# print(test_x[test_y != pred_test_y])

# 画图：
mp.figure('SVM Classification', facecolor='lightgray')
mp.title('SVM Classification', fontsize=16)

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

mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y,
           cmap='brg_r', label='Test Samples', s=80)
mp.legend()
mp.show()
