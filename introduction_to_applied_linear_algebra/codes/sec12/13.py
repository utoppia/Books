#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 12.13
# Author: Utoppia
# Date  : 13 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt

def main():
    m, n = 20, 10
    N = 500

    A = np.random.rand(m, n)
    b = np.random.rand(m)

    A_pseudo = pinv(A)
    x_hat = A_pseudo.dot(b)

    mu = 1/(norm(A)**2)
    F = np.identity(10) - mu * (A.T.dot(A))
    g = mu * A.T.dot(b)
    x = np.zeros(10)

    residual = np.zeros(N+1)
    residual[0] = norm(x-x_hat)
    for i in range(N):
        x = F.dot(x) + g
        residual[i+1] = norm(x-x_hat)

    plt.plot(residual)
    plt.xlabel('Iteration Number')
    plt.ylabel('Residual Error')
    plt.title('The $||x^{(k)}-\hat{x}||$ in Richardson algorithm for exercise 12.13')
    plt.grid(True)
    plt.savefig('12-13.pdf')

if __name__ == '__main__':
    main()