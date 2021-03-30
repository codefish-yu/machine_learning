# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_vc.py   捕获视频
"""
import cv2 as cv

vc = cv.VideoCapture(0)

while True:
    frame = vc.read()[1]
    cv.imshow('frame', frame)
    if cv.waitKey(33) == 27:
        break

# 释放资源
vc.release()
cv.destroyAllWindows()
