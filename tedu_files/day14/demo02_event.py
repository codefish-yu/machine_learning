# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_event.py  事件预测
"""

import numpy as np
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm


class DigitEncoder():

    def fit_transform(self, array):
        return array.astype(int)

    def transform(self, array):
        return array.astype(int)

    def inverse_transform(self, array):
        return array.astype(str)


# 读取文本
data = np.loadtxt(
    '../ml_data/events.txt', delimiter=',',
    dtype='U20')
data = np.delete(data, 1, axis=1)
print(data.shape, data[0])
# 整理输入集与输出集
x, y, encoders = [], [], []
for i in range(data.shape[1]):
    col_val = data[:, i]  # 获取当前列数组
    # 根据当前列的内容，判断使用哪一个encoder
    if col_val[0].isdigit():
        encoder = DigitEncoder()
    else:
        encoder = sp.LabelEncoder()
    # 对当前列进行编码
    if i < data.shape[1] - 1:
        x.append(encoder.fit_transform(col_val))
    else:
        y = encoder.fit_transform(col_val)
    encoders.append(encoder)

x = np.array(x).T
y = np.array(y)
print(x.shape, y.shape, x[0], y[0])

# 训练模型
train_x, test_x, train_y, test_y = \
    ms.train_test_split(
        x, y, test_size=0.1, random_state=7)
model = svm.SVC(
    kernel='rbf', class_weight='balanced')
model.fit(train_x, train_y)
# 测试
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))

# 应用
data = np.array([['Tuesday', '16:30:00', '21', '23']])
# 对每一列使用相同的编码器进行编码，然后再预测
x = []
for i in range(data.shape[1]):
    col_val = data[:, i]  # 获取当前列数组
    # 对当前列进行编码
    encoder = encoders[i]
    x.append(encoder.transform(col_val))
test_x = np.array(x).T
pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(pred_test_y))
