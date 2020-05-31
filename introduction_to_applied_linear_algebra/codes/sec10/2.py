#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 10.30
# Author: Utoppia
# Date  : 08 May 2020

import numpy as np 
from numpy.linalg import matrix_power

def main():

    A = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ])

    print(matrix_power(A, 10))

main()