#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 16.5
# Author: Utoppia
# Date  : 27 May 2020

import numpy as np 
from numpy import array, eye, zeros, linspace, sqrt
from numpy.linalg import pinv, norm

def least_square_constrained(A, b, C, d):
    """
    Solve the linear constrained least squares problem:
        minimize    ||Ax - b||^2
        subject to  Cx = d
    let x' = (x, z), z is the Langrange multipliers of constraits equalities, 
    problem above is same to solve 
        | 2A^TA C^T | | x |   | 2A^Tb |
        |           | |   | = |       |
        | C     0   | | z |   | d     |
    ---------------------------------------
    Parameters:
    A : matrix 
    b : array-like
    C : matrix 
    d : array-like
    """
    _, n = A.shape
    p, _ = C.shape 
   
    M = np.block([
        [2 * A.T @ A, C.T],
        [C, zeros((p,p))]
    ])
    N = np.hstack([2 * A.T @ b, d])

    X = pinv(M) @ N
    x, z = X[:n], X[n:]
    return x, z

# generate (20, 10) matrix A
A = np.random.random((20,10))
# generate (5,10) matrix C
C = np.random.random((5,10))
# generate b and d
b = np.random.random(20)
d = np.random.random(5)

x, z = least_square_constrained(A, b, C, d)

print ("Cx-d is")
print(C @ x - d)
print(norm(C @ x -d))


x2 = pinv(C) @ d 
print(x2)
print("Cx2-d is")
print(C @ x2 - d)
print(norm(C @ x2 -d))

print(norm(A @ x2 - b ))
print(norm(A @ x -b)) 