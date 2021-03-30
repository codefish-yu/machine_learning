# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_sigmoid.py
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(-50, 50, 500)
y = 1 / (1 + np.exp(-x))

mp.plot(x, y)
mp.grid(linestyle=':')
mp.show()
