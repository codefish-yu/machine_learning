# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_canny.py  边缘识别
"""
import cv2 as cv

original = cv.imread('../ml_data/chair.jpg', 0)
print(original.shape)
cv.imshow('original', original)
# Sobel
hsobel = cv.Sobel(original, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('hsobel', hsobel)
vsobel = cv.Sobel(original, cv.CV_64F, 0, 1, ksize=5)
cv.imshow('vsobel', vsobel)
sobel = cv.Sobel(original, cv.CV_64F, 1, 1, ksize=5)
cv.imshow('sobel', sobel)
# 拉普拉斯边缘识别
laplacian = cv.Laplacian(original, cv.CV_64F)
cv.imshow('Laplacian', laplacian)
# Canny
canny = cv.Canny(original, 50, 240)
cv.imshow('canny', canny)

cv.waitKey()
