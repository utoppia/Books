#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 16.12
# Author: Utoppia
# Date  : 22 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
import math

def main():
    A = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4]
    ])
    b = np.array([0, 0, 1, 0]).reshape(-1,1)
    c = pinv(A).dot(b).reshape(1,-1)[0]
    
    # print
    for i in range(5):
        print('c{} = {:.4f}'.format(i+1, c[i]))

    # plot
    x = np.linspace(-1,1,100)
    f = np.frompyfunc(lambda x: c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3 + c[4]*x**4, 1, 1)
    y = f(x)

    f_der = np.frompyfunc(lambda x: c[1] + 2*c[2]*x + 3*c[3]*x**2 + 4*c[4]*x**3, 1, 1)
    y_der = f_der(x)

    fig = plt.figure()
    plt.plot(x, y, label='polynomial curve')
    plt.plot(x, y_der, label='derivative curve')

    x = np.array([0, 1])
    y = f(x)
    y_der = f_der(x)
    plt.scatter(x, y, color='red')
    plt.scatter(x, y_der, color='red')

    plt.legend()
    plt.grid(True)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    fig.savefig('16-12.pdf')

if __name__ == '__main__':
    main()
