#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 11.21
# Author: Utoppia
# Date  : 12 May 2020

import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt

def main():
    t = np.array([-0.6,-0.2,0.2,0.6])
    b = np.exp(t)
    A = np.array([
        t**0,
        t**1,
        t**2,
        t**3
    ])
    print('A is:'); print(A)
    print('b is:'); print(b)
    print('Inverse of A is:'); print(inv(A))
    w = inv(A).dot(b)
    print('w is'); print(w)

    alpha = np.e - 1.0/np.e
    hat_alpha = w.dot(b)
    print('alpha is ', alpha)
    print('hat_alpha is ', hat_alpha)

if __name__ == '__main__':
    main()