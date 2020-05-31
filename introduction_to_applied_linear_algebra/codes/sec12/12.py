#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 12.12
# Author: Utoppia
# Date  : 12 May 2020

import numpy as np 
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt


def main():
    N, K, L = 10, 4, 13
    edges = [
        (1,3), (1,4), (1,7), (2,3), (2,5), (2,8), (2,9),
        (3,4), (3,5), (4,6), (5,6), (6,9), (6,10)
    ]
    B = np.zeros((N, L))
    for (i, (u,v)) in enumerate(edges):
        B[u-1][i] = 1
        B[v-1][i] = -1

    print('The incidence matrix B is')
    print(B)

    C = B[:N-K,:]
    D = B[N-K:,:]
    print('C is')
    print(C)
    print('D is')
    print(D)
    
    A = np.block([
        [C.T, np.zeros((L, N-K))],
        [np.zeros((L, N-K)), C.T]
    ])
    print('A is')
    print(A)

    u = np.array([0,0,1,1,0,1,1,0])
    b = - np.block([
        [D.T, np.zeros((L, K))],
        [np.zeros((L,K)), D.T]
    ]).dot(u)
    print('b is')
    print(b)

    x = pinv(A).dot(b)
    print(x)

    p = np.block([
        [x[:N-K], u[:K]], 
        [x[N-K:], u[K:]]
    ])
    print(p)
    plt.scatter(x[:N-K], x[N-K:], color='red', marker='s')
    plt.scatter(u[:K], u[K:], color='black', marker='o')

    for (u,v) in edges:
        plt.plot(p[0][[u-1, v-1]], p[1][[u-1,v-1]], color='blue')
    
    adj = np.array([
        [0, 0, -0.05, 0, 0, 0.05, 0, 0, 0, 0],
        [-0.05, 0.05, 0, -0.05, -0.05, 0, 0.05, -0.05, -0.05, 0.05]
    ])
    for i in range(N):
        plt.text(p[0][i]+adj[0][i], p[1][i]+adj[1][i], '${}$'.format(i+1))
    
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Least squares placement for exercise 12.12')
    plt.savefig('12-12.pdf')
    
if __name__ == '__main__':
    main()