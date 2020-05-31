#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 16.10
# Author: Utoppia
# Date  : 21 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
import math


def solver(A, b, C, d):
    """
    Solution of constrained least problem:
        minimize     ||Ax + b||^2
        subbject to  Cx = d
       
    The stacked matrix is 
        
        | 2A^TA   C^T |   | x |   | 2A^Tb |
        |             | * |   | = |       |
        | C       0   |   | z |   | d     |
    """

    # create the stacked matrix
    
    m  = C.shape[0]
    P = np.block([
        [2*A.T.dot(A), C.T],
        [C, np.zeros((m,m))]
    ])
    B = np.block([
        [2*A.T.dot(b)],
        [d]
    ])
    y = pinv(P).dot(B).reshape(1,-1)[0]
    return y[:10], y[10:]

def velocity(t, f):
    """
    Parameters:
        t -> float: time
        f -> array like: force sequence
    """
    if t==0:
        return 0
    K = math.ceil(t)
    return np.sum(f[:K-1]) + (t-K+1)*f[K-1]
    
def position(t, f):
    """
    Parameters:
        t -> float: time
        f -> array like: force sequence
    """
    if t == 0:
        return 0
    K = math.ceil(t)
    ans = 0
    for i in range(K-1):
        ans += (2*K-2*i-1)*f[i]/2.0
    ans += (t-K)*np.sum(f[:K-1])
    ans += (t-K+1)**2 * f[K-1] / 2
    return ans

def plot(f1, f2):
    x = np.linspace(0, 10, 100)
    v1 = [ velocity(t, f1) for t in x ]
    p1 = [ position(t, f1) for t in x ]
    v2 = [ velocity(t, f2) for t in x ]
    p2 = [ position(t, f2) for t in x ]
    fig = plt.figure()

    plt.plot(x, p1, label='smallest force')
    plt.plot(x, p2, label="specific position")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Position')
    fig.savefig('16-10-a.pdf')

    fig = plt.figure()
    plt.plot(x, v1, label='smallest force')
    plt.plot(x, v2, label='specific position')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    fig.savefig('16-10-b.pdf')

    
    x = []
    y1, y2 = [], []
    for i in range(10):
        x.append(i); y1.append(f1[i]); y2.append(f2[i])
        x.append(i+1); y1.append(f1[i]); y2.append(f2[i])
    
    fig=plt.figure()
    plt.plot(x,y1, label="smallest force")
    plt.plot(x, y2, label='specific position')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Force')
    fig.savefig('16-10-c.pdf')


def main():
    A = np.identity(10)
    b = np.zeros((10, 1))
    C = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [19/2, 17/2, 15/2, 13/2, 11/2, 9/2, 7/2, 5/2, 3/2, 1/2]
    ])
    d = np.array([
        [0],
        [1]
    ])

    
    f1, _ = solver(A, b, C, d)
    
    C = np.array([
        [19/2, 17/2, 15/2, 13/2, 11/2, 9/2, 7/2, 5/2, 3/2, 1/2] 
    ])
    d = np.array([[1]])

    f2, _ = solver(A, b, C, d)
    plot(f1, f2)

    print("norm of smallest force is ", norm(f1))
    print("norm of specific position is ", norm(f2))
    
if __name__ == '__main__':
    main()
