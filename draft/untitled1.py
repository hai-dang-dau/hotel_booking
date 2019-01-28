#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:43:20 2019

@author: haidang
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

x = ['s0'] * 100
y = list(np.random.choice(['l0', 'l1'], size = 100, p = [0.7, 0.3]))
x += ['s1'] * 500
y += list(np.random.choice(['l0', 'l1'], size = 500, p = [0.5, 0.5]))
df = pd.DataFrame({'tlx': x, 'tly':y})

ct = pd.crosstab(index = df.tlx, columns = df.tly)
ct = ct.mul(1 / ct.sum(axis = 1), axis = 0)
ax = ct.plot(kind = "bar", rot = 0)
ax.set(xlabel = "fx", ylabel = "fy")
ax