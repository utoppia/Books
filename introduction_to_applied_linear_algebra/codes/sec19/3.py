#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exercise 19.3
# Author: Utoppia
# Date  : 31 May 2020
from numpy import array, random, eye, ones, hstack, vstack, diag, zeros
from numpy.linalg import norm, pinv
from utils import levenberg_marquadt, augmented_lagrangian
from math import sqrt

def sign(x):
    return array([1 if item > 0 else -1 for item in x])

def brute(A, b, n):
    def split(x):
        ret = []
        for _ in range(n):
            ret.append(x%2)
            x = x//2
        return ret
    res, x = 10**10, None
    for i in range(2**n):
        q = split(i)
        xk = sign(q)

        res_k = norm(A @ xk - b)**2
        if res_k < res:
            res = res_k
            x = xk
    return x, res 

def solver(A, b):
    f = lambda x: A @ x - b
    Df = lambda x: A 
    g = lambda x: x**2 - 1
    Dg = lambda x: 2 * diag(x)
    x0 = pinv(A) @ b # x0 to be the minimize of ||Ax-b||^2 
    x0 = sign(x0)
    z0 = ones(n)
    x, z, _, history = augmented_lagrangian(x0, z0, f, Df, g, Dg)
    return sign(x), norm(A @ sign(x) - b)**2


for _ in range(100):
    n, m = 10, 10
    A = random.random((m,m))
    b = random.random(m)

    #x, res = brute(A, b, n)
    #print('Brute force')
    #print(x, res)
    x, res = solver(A, b)
    print('Augmented Lagrangian')
    print(x, res)
