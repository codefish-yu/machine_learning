# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_quant.py 图像量化
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.cluster as sc
import scipy.misc as sm

img = sm.imread('../ml_data/lily.jpg', True)
print(img.shape)

# 图像量化
x = img.reshape(-1, 1)
model = sc.KMeans(n_clusters=4)
model.fit(x)
y = model.labels_  # 262144个元素的一维数组
centers = model.cluster_centers_  # (4, 1)聚类中心
print(y.shape)
print(centers)
img2 = centers[y].reshape(img.shape)

mp.subplot(121)
mp.imshow(img, cmap='gray')
mp.subplot(122)
mp.imshow(img2, cmap='gray')
mp.show()
