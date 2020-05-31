#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Section 19
# Author: Utoppia
# Date  : 31 May 2020

import numpy as np 
from numpy import array
from utils import levenberg_marquadt, augmented_lagrangian, penalty
from math import exp

f = lambda x: array([x[0] + exp(-x[1]), x[0]*x[0]+2*x[1]+1])
Df = lambda x: array([
    [1, -exp(-x[1])],
    [2*x[0], 2]
])

g = lambda x: x[0] + x[0]**3 + x[1] + x[1]**2 
Dg = lambda x: array([
    [1 + 3 * x[0]**2,1 + 3 * x[1]**2]
])

x0 = (0.5, -0.5) 
z0 = array([1])

x, _, = levenberg_marquadt(x0, f, Df)
print(x)

x1 = penalty(x0, f, Df, g, Dg)
print(x1)

x1, z, status, history = augmented_lagrangian(x0, z0, f, Df, g, Dg)
print(x1, z)
print(history['penalty'])