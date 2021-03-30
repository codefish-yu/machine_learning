# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_kmeans.py  kmeans聚类
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.cluster as sc

x = np.loadtxt(
    '../ml_data/multiple3.txt', delimiter=',')
print(x.shape)

# 训练聚类模型，划分4个聚类
model = sc.KMeans(n_clusters=4)
model.fit(x)
y = model.labels_

# 输出轮廓系数
import sklearn.metrics as sm
score = sm.silhouette_score(
    x, y, sample_size=len(x), metric='euclidean')
print(score)

centers = model.cluster_centers_
print(centers)

mp.figure('KMeans Cluster', facecolor='lightgray')
mp.title('KMeans Cluster', fontsize=16)

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

mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg',
           s=80, label='Samples', alpha=0.8)
mp.scatter(centers[:, 0], centers[:, 1],
           color='yellow', marker='+',
           s=1000, label='centers')
mp.legend()
mp.show()
