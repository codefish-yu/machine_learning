# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_opencv.py 
pip3 install opencv-python
pip3 install opencv-contrib-python
"""
import cv2 as cv
import numpy as np

original = cv.imread('../ml_data/forest.jpg')
print(original.shape, original[0, 0])
cv.imshow('original', original)
# 切出某通道的数据
blue = np.zeros_like(original)
blue[:, :, 0] = original[:, :, 0]
cv.imshow('blue', blue)

green = np.zeros_like(original)
green[:, :, 1] = original[:, :, 1]
cv.imshow('green', green)

red = np.zeros_like(original)
red[:, :, 2] = original[:, :, 2]
cv.imshow('red', red)

# 图像裁剪
h, w = original.shape[:2]
t, b = int(h / 4), int(h * 3 / 4)
l, r = int(w / 4), int(w * 3 / 4)
cropped = original[t:b, l:r]
cv.imshow('cropped', cropped)

# 图像缩放
resize1 = cv.resize(original, (150, 100))
cv.imshow('resize1', resize1)

resize2 = cv.resize(resize1, None, fx=4, fy=4)
cv.imshow('resize2', resize2)

# 图像保存
cv.imwrite('red.png', red)

cv.waitKey()
