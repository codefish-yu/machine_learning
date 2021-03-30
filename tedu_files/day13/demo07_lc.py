# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_lc.py  学习曲线
"""
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms
import matplotlib.pyplot as mp

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
    max_depth=9, n_estimators=140, random_state=7)

# 使用学习曲线，选择最优训练比例
train_sizes = np.linspace(0.1, 1.0, 10)
_, train_scores, test_scores = ms.learning_curve(
    model, train_x, train_y,
    train_sizes=train_sizes, cv=5)
mp.figure('Learning Curve', facecolor='lightgray')
mp.title('Learning Curve', fontsize=16)
mp.grid(linestyle=':')
mp.plot(train_sizes,
        test_scores.mean(axis=1), 'o-',
        color='dodgerblue', label='Learning Curve')
mp.legend()
mp.show()

# # 验证曲线选择最优max_depth
# train_scores, test_scores = ms.validation_curve(
#     model, train_x, train_y,
#     'max_depth', np.arange(1, 11), cv=5)
# print(test_scores.mean(axis=1))

# mp.figure('Validation Curve', facecolor='lightgray')
# mp.title('Validation Curve', fontsize=16)
# mp.grid(linestyle=':')
# mp.plot(np.arange(1, 11),
#         test_scores.mean(axis=1), 'o-',
#         color='dodgerblue', label='Validation Curve')
# mp.legend()
# mp.show()

# # 验证曲线选择最优n_estimators
# train_scores, test_scores = ms.validation_curve(
#     model, train_x, train_y,
#     'n_estimators', np.arange(100, 210, 10), cv=5)
# print(test_scores.mean(axis=1))

# mp.figure('Validation Curve', facecolor='lightgray')
# mp.title('Validation Curve', fontsize=16)
# mp.grid(linestyle=':')
# mp.plot(np.arange(100, 210, 10),
#         test_scores.mean(axis=1), 'o-',
#         color='dodgerblue', label='Validation Curve')
# mp.legend()
# mp.show()

model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)
import sklearn.metrics as sm
print(sm.classification_report(train_y, pred_train_y))

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
