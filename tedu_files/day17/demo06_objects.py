# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_objects.py 图像识别
"""
import numpy as np
import cv2 as cv
import os
import sklearn.svm as svm
import sklearn.metrics as sm


def search_files(directory):
    # 通过directory 加载所有文件，存入字典返回
    # {'apple':[url, url....], 'banana':[url ...]}
    objects = {}
    for curdir, subdir, files in os.walk(directory):
        for filename in files:
            path = os.path.join(curdir, filename)
            label = curdir.split(os.path.sep)[-1]
            if label not in objects.keys():
                objects[label] = []
            objects[label].append(path)
    return objects

files = search_files('../ml_data/objects/training')
print(files)
# 重新读取files字典中的内容，整理输入与输出
train_x, train_y = [], []
for label, urls in files.items():
    # 读取urls中的每个音频文件
    for url in urls:
        # 图像压缩
        img = cv.imread(url, 0)
        h, w = img.shape[:2]
        scale = 200 / max(h, w)
        gray = cv.resize(img, None, fx=scale, fy=scale)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints = sift.detect(gray)
        _, desc = sift.compute(gray, keypoints)
        desc = desc.mean(axis=0)
        train_x.append(desc)
        train_y.append(label)
train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x.shape, train_y.shape)
# 针对train_y做标签编码，把字符串变成数字
import sklearn.preprocessing as sp
lbe = sp.LabelEncoder()
train_y = lbe.fit_transform(train_y)

# 训练模型
model = svm.SVC(kernel='linear', probability=True)
model.fit(train_x, train_y)
pred_train_y = model.predict(train_x)
print(sm.classification_report(train_y, pred_train_y))

# 基于测试数据进行测试
files = search_files('../ml_data/objects/testing')
# 重新读取files字典中的内容，整理输入与输出
test_x, test_y = [], []
for label, urls in files.items():
    # 读取urls中的每个音频文件
    for url in urls:
        # 图像压缩
        img = cv.imread(url, 0)
        h, w = img.shape[:2]
        scale = 200 / max(h, w)
        gray = cv.resize(img, None, fx=scale, fy=scale)
        sift = cv.xfeatures2d.SIFT_create()
        keypoints = sift.detect(gray)
        _, desc = sift.compute(gray, keypoints)
        desc = desc.mean(axis=0)
        test_x.append(desc)
        test_y.append(label)
test_x = np.array(test_x)
test_y = lbe.transform(np.array(test_y))
pred_test_y = model.predict(test_x)
print(sm.classification_report(test_y, pred_test_y))
probs = model.predict_proba(test_x)
# print(probs.max(axis=1))

for ty, pty, prop in zip(lbe.inverse_transform(test_y),
                         lbe.inverse_transform(pred_test_y),
                         probs.max(axis=1)):
    print(ty, '->', pty, ' : ', prop)
