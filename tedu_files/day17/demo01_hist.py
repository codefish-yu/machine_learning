# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_hist.py 直方图均衡化提亮
"""
import cv2 as cv

original = cv.imread('../ml_data/sunrise.jpg')
cv.imshow('original', original)

gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

egray = cv.equalizeHist(gray)
cv.imshow('egray', egray)

# BGR YUV
yuv = cv.cvtColor(original, cv.COLOR_BGR2YUV)
# 为亮度通道做直方图均衡化
yuv[:, :, 0] = cv.equalizeHist(yuv[:, :, 0])
color = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
cv.imshow('color', color)

cv.waitKey()
