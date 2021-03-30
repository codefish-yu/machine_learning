# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_fi.py 特征重要性
"""
import numpy as np
import sklearn.datasets as sd
import sklearn.tree as st
import sklearn.utils as su
import sklearn.metrics as sm
import matplotlib.pyplot as mp

boston = sd.load_boston()
print(boston.data.shape, boston.data[0])
print(boston.target.shape, boston.target[0])
print(boston.feature_names)
header = boston.feature_names

# 打乱数据集，拆分训练集与测试集
x, y = su.shuffle(
    boston.data, boston.target, random_state=7)
# 测试集训练集的拆分
train_size = int(len(x) * 0.8)
train_x, test_x, train_y, test_y = \
    x[:train_size], x[train_size:],\
    y[:train_size], y[train_size:]
# 训练模型
model = st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
# 预测
pred_test_y = model.predict(test_x)
# 评估
print(sm.r2_score(test_y, pred_test_y))

# 输出单棵决策树模型的特征重要性
fi = model.feature_importances_
mp.figure('Feature Importance', facecolor='lightgray')
mp.subplot(211)
mp.title('DT FI', fontsize=16)
mp.grid(linestyle=':', axis='y')
x = np.arange(len(fi))
# 排序
sorted_ind = np.argsort(fi)[::-1]
mp.xticks(x, header[sorted_ind])
mp.bar(x, fi[sorted_ind], 0.8, color='dodgerblue',
       label='DT')
mp.legend()
mp.tight_layout()

# 正向激励
import sklearn.ensemble as se
model = st.DecisionTreeRegressor(max_depth=4)
model = se.AdaBoostRegressor(
    model, n_estimators=400, random_state=7)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
fi = model.feature_importances_
mp.subplot(212)
mp.title('Adaboost FI', fontsize=16)
mp.grid(linestyle=':', axis='y')
x = np.arange(len(fi))
# 排序
sorted_ind = np.argsort(fi)[::-1]
mp.xticks(x, header[sorted_ind])
mp.bar(x, fi[sorted_ind], 0.8, color='orangered',
       label='Adaboost')
mp.legend()
mp.tight_layout()
mp.show()
