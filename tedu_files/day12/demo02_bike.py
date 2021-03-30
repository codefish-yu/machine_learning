# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_bike.py  预测共享单车投放量
"""
import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp

# 加载数据
data = []
with open('../ml_data/bike_day.csv', 'r') as f:
    for line in f.readlines():
        data.append(line[:-1].split(','))
data = np.array(data)
# 整理header，输入集与输出集
header = data[0, 2:13]
x = data[1:, 2:13].astype('f8')
y = data[1:, -1].astype('f8')
print(x.shape, y.shape)

# 打乱数据集， 拆分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:]
# 构建随机森林回归模型
model = se.RandomForestRegressor(
    max_depth=10, n_estimators=1000,
    min_samples_split=2)
model.fit(train_x, train_y)
# 测试评估模型
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))

# 输出单棵决策树模型的特征重要性
fi = model.feature_importances_
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('Bike Day', fontsize=16)
mp.grid(linestyle=':', axis='y')
x = np.arange(len(fi))
# 排序
sorted_ind = np.argsort(fi)[::-1]
mp.xticks(x, header[sorted_ind])
mp.bar(x, fi[sorted_ind], 0.8, color='dodgerblue',
       label='Bike Day')
mp.legend()
mp.tight_layout()


# 加载数据
data = []
with open('../ml_data/bike_hour.csv', 'r') as f:
    for line in f.readlines():
        data.append(line[:-1].split(','))
data = np.array(data)
# 整理header，输入集与输出集
header = data[0, 2:14]
x = data[1:, 2:14].astype('f8')
y = data[1:, -1].astype('f8')
print(x.shape, y.shape)

# 打乱数据集， 拆分测试集与训练集
x, y = su.shuffle(x, y, random_state=7)
train_size = int(len(x) * 0.9)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:], \
    y[:train_size], y[train_size:]
# 构建随机森林回归模型
model = se.RandomForestRegressor(
    max_depth=10, n_estimators=1000,
    min_samples_split=2)
model.fit(train_x, train_y)
# 测试评估模型
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
fi = model.feature_importances_
mp.subplot(212)
mp.title('Bike Hour', fontsize=16)
mp.grid(linestyle=':', axis='y')
x = np.arange(len(fi))
# 排序
sorted_ind = np.argsort(fi)[::-1]
mp.xticks(x, header[sorted_ind])
mp.bar(x, fi[sorted_ind], 0.8, color='orangered',
       label='Hour')
mp.legend()
mp.tight_layout()
mp.show()
