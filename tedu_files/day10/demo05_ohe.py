# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_ohe.py  独热编码
"""
import numpy as np
import sklearn.preprocessing as sp

data = np.mat('1 3 2; 7 5 4; 1 8 6; 7 3 9')
print(data)

# 独热编码
ohe = sp.OneHotEncoder()
r = ohe.fit_transform(data)
print(r, type(r))
print(r.toarray())
