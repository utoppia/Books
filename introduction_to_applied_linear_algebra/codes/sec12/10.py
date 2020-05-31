#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 12.10
# Author: Utoppia
# Date  : 29 May 2020

import numpy as np 
from numpy.linalg import pinv, norm

# generate (30, 10) A 
A = np.random.random((30,10))
# generate (30, ) vector b
b = np.random.random(30)

x = pinv(A) @ b 

residual = norm(A @ x - b) 

d = [np.random.random(10) for _ in range(3)]

for i in range(3):
    print(norm(A @ (x + d[i]) - b) > residual)
