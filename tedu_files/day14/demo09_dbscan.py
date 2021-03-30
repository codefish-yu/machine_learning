# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_dbscan.py 
"""
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.metrics as sm
x = np.loadtxt('../ml_data/perf.txt',
               delimiter=',')

# 选择最优半径
eps, scores, models = np.linspace(0.3, 1.2, 10), [], []
for r in eps:
    model = sc.DBSCAN(eps=r, min_samples=3)
    model.fit(x)
    y = model.labels_
    score = sm.silhouette_score(
        x, y, sample_size=len(x), metric='euclidean')
    scores.append(score)
    models.append(model)

best_ind = np.argmax(scores)  # 分最高的索引
best_model = models[best_ind]
best_score = scores[best_ind]
best_r = eps[best_ind]

mp.figure('DBSCAN Cluster', facecolor='lightgray')
mp.title('DBSCAN Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
# 设置每个核心样本的颜色
core_ind = best_model.core_sample_indices_
labels = best_model.labels_
core_mask = labels == -2
core_mask[core_ind] = True
mp.scatter(x[:, 0][core_mask], x[:, 1][core_mask],
           c=labels[core_mask], cmap='brg', s=80)

# 设置每个孤立样本的颜色
offset_mask = labels == -1
mp.scatter(x[:, 0][offset_mask], x[:, 1][offset_mask],
           c='gray', marker='D', s=100, alpha=0.6)

# 设置每个外周样本的颜色
p_mask = ~(core_mask | offset_mask)
mp.scatter(x[:, 0][p_mask], x[:, 1][p_mask], marker='*',
           c=labels[p_mask], cmap='brg', s=80, alpha=0.6)


mp.show()
