# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_thres.py 二值化
"""
import numpy as np
import sklearn.preprocessing as sp
import scipy.misc as sm
import matplotlib.pyplot as mp

img = sm.imread('../da_data/lily.jpg', True)
print(img.shape)
mp.subplot(121)
mp.imshow(img, cmap='gray')
mp.tight_layout()

# 针对图像执行二值化处理，简化图像处理难度
bin = sp.Binarizer(threshold=127)
img2 = bin.transform(img)
mp.subplot(122)
mp.imshow(img2, cmap='gray')
mp.tight_layout()

mp.show()
