# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_gridsearch.py  网格搜索
"""
import numpy as np
import matplotlib.pyplot as mp
import sklearn.model_selection as ms
import sklearn.svm as svm

# 读取样本数据
data = np.loadtxt(
    '../ml_data/multiple2.txt', delimiter=',')
x = data[:, :-1]
y = data[:,  -1]
print(x.shape, y.shape)

# 拆分训练集与测试集
train_x, test_x, train_y, test_y = \
    ms.train_test_split(
        x, y, test_size=0.25, random_state=7)

# 训练svm模型
model = svm.SVC(probability=True)

# 基于网格搜索寻求最优超参数组合
params = [
    {'kernel': ['linear'], 'C':[1, 10, 100, 1000]},
    {'kernel': ['poly'], 'C':[1], 'degree':[2, 3]},
    {'kernel': ['rbf'], 'C':[1, 10, 100, 1000],
     'gamma':[1, 0.1, 0.01, 0.001]}]
model = ms.GridSearchCV(model, params, cv=5)
model.fit(train_x, train_y)
# 训练过后，获取网格搜索最优超参数组合
print(model.best_params_)
print(model.best_score_)
print(model.best_estimator_)
# 输出交叉验证过程中所有模型的得分：
for p, s in zip(
        model.cv_results_['params'],
        model.cv_results_['mean_test_score']):
    print(p, s)

# 针对测试集样本进行预测
pred_test_y = model.predict(test_x)
print('acc:', (test_y == pred_test_y).sum() / test_y.size)
# 输出错误样本的特征数据
# print(test_x[test_y != pred_test_y])

# 画图：
mp.figure('SVM Classification', facecolor='lightgray')
mp.title('SVM Classification', fontsize=16)

# 绘制分类边界线
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
n = 500
grid_x, grid_y = np.meshgrid(
    np.linspace(l, r, n), np.linspace(b, t, n))
# 预测每个点的类别标签
grid_z = model.predict(np.column_stack(
    (grid_x.ravel(), grid_y.ravel())))
grid_z = grid_z.reshape(grid_x.shape)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')

mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y,
           cmap='brg_r', label='Test Samples', s=80)

# 整理一组新数据，输出它们的置信概率
prob_x = np.array([
    [2, 1.5],
    [8, 9],
    [4.8, 5.2],
    [4, 4],
    [2.5, 7],
    [7.6, 2],
    [5.4, 5.9]])
prob_y = model.predict(prob_x)
mp.scatter(prob_x[:, 0], prob_x[:, 1], c=prob_y,
           marker='D', s=100,
           cmap='jet', label='Prob Samples')
probs = model.predict_proba(prob_x)
print(probs)
for i in range(len(probs)):
    mp.annotate(
        '{}% {}%'.format(
            round(probs[i, 0] * 100, 2),
            round(probs[i, 1] * 100, 2)),
        xy=(prob_x[i, 0], prob_x[i, 1]),
        xytext=(12, -12),
        textcoords='offset points',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=9,
        bbox={'boxstyle': 'round,pad=0.6',
              'fc': 'orange', 'alpha': 0.8})

mp.legend()
mp.show()
