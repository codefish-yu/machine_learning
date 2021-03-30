# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_adaboost.py 预测波士顿房屋价格
"""
import numpy as np
import sklearn.datasets as sd
import sklearn.tree as st
import sklearn.utils as su
import sklearn.metrics as sm

boston = sd.load_boston()
print(boston.data.shape, boston.data[0])
print(boston.target.shape, boston.target[0])
print(boston.feature_names)

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
print(sm.mean_absolute_error(test_y, pred_test_y))

# 正向激励
import sklearn.ensemble as se
model = st.DecisionTreeRegressor(max_depth=4)
model = se.AdaBoostRegressor(
    model, n_estimators=400, random_state=7)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
print(sm.mean_absolute_error(test_y, pred_test_y))


# 线性回归
import sklearn.linear_model as lm

Cs = np.arange(0, 400, 10)
for C in Cs:
    model = lm.Ridge(C, fit_intercept=True)
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)
    print(C, ':', sm.r2_score(test_y, pred_test_y))
