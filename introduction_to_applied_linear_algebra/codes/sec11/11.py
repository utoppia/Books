#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 11.11
# Author: Utoppia
# Date  : 11 May 2020

import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt

def main():
    A = np.array([
        [1, 1, 1, -2, -2 ],
        [1, 2, 4, -10, -20 ],
        [1, 3, 9, -27, -81 ],
        [1, 4, 16, 4, 16 ],
        [1, 5, 25, 20, 100]
    ])
    b = np.array([2, 5, 9, -1, -4])

    a = inv(A)
    print('Invese of A is:')
    print(a)

    x = a.dot(b)
    print('solution of equations x =')
    print(x)
    
    return a, x

def f(x, w):
    return (w[0] + w[1]*x + w[2]*(x**2)) / (1 + w[3]*x + w[4]*(x**2))

def plot(w):
    x = np.linspace(0,6,100)
    y = f(x, w)
    plt.plot(x, y)

    x1 = np.array([1,2,3,4,5])
    y1 = f(x1, w)
    plt.scatter(x1, y1, color='r')
    
    plt.yticks(np.linspace(-5,10,16))
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('The plot of exercise 11.11')
    plt.grid(True)
    plt.savefig('ex11-11.pdfpw')

if __name__ == '__main__':
    a, x = main()
    plot(x)