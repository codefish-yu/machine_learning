# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_harris.py  harris角点检测
"""
import cv2 as cv

original = cv.imread('../ml_data/box.png')
cv.imshow('Original', original)
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
corners = cv.cornerHarris(gray, 7, 5, 0.04)
print(gray.shape, corners.shape)
corners[corners > corners.max() * 0.01] = 255
cv.imshow('corners', corners)

mixture = original.copy()
mixture[corners > corners.max() * 0.01] = [0, 0, 255]
cv.imshow('Corner', mixture)
cv.waitKey()
