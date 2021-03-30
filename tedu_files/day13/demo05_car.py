# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_car.py  小汽车评级
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

# 加载样本数据，整理输入与输出
data = []
with open('../ml_data/car.txt', 'r') as f:
    for line in f.readlines():
        data.append(line[:-1].split(','))
data = np.array(data)
# 读取data中的每一列，完成标签编码
train_x, train_y, encoders = [], [], []
for i in range(data.shape[1]):
    col = data[:, i]
    lbe = sp.LabelEncoder()
    col_coded = lbe.fit_transform(col)
    # 通过判断是否是最后一列数据，整理输入与输出集
    if i < data.shape[1] - 1:
        train_x.append(col_coded)
    else:
        train_y = col_coded
    encoders.append(lbe)

train_x = np.array(train_x).T
train_y = np.array(train_y)
print(train_x[0], train_y[0])

# 开始训练模型
model = se.RandomForestClassifier(
    max_depth=6, n_estimators=200, random_state=7)
# 交叉验证
score = ms.cross_val_score(
    model, train_x, train_y, cv=5, scoring='f1_weighted')
print(score.mean())
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)
# import sklearn.metrics as sm
# print(sm.classification_report(train_y, pred_train_y))

# 应用
data = np.array([
    ['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
    ['high', 'high', '4', '4', 'med', 'med', 'acc'],
    ['low', 'low', '2', '4', 'small', 'high', 'good'],
    ['low', 'med', '3', '4', 'med', 'high', 'vgood']])
test_x, test_y = [], []
for i in range(data.shape[1]):
    col = data[:, i]
    lbe = encoders[i]  # 获取当时训练时的Encoder
    col_coded = lbe.transform(col)
    # 通过判断是否是最后一列数据，整理输入与输出集
    if i < data.shape[1] - 1:
        test_x.append(col_coded)
    else:
        test_y = col_coded
test_x = np.array(test_x).T
test_y = np.array(test_y)
# 预测
pred_test_y = model.predict(test_x)
print(encoders[-1].inverse_transform(test_y))
print(encoders[-1].inverse_transform(pred_test_y))
