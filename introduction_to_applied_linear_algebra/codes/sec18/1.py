#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.1
# Author: Utoppia
# Date  : 23 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
import math

def f(u):
    return lambda x: x * (math.exp(x)) - u

def f_diff(u):
    return lambda x: math.exp(x) + x * (math.exp(x)) - u

def lambert_marquardt(x_init, f, f_diff, lbd=1, err=10**(-6), nMax = 10000):
    """
    Lambert-Marquardt algorithm to solve nonlinear least squares problem,
    for 1 dimension variable
    -------------------
    Parameters:
        x_init : Float. Initialization of the variable with 1 dimension
        f : calable function 
        f_diff: calable function, the differential coefficient of f
        lbd: Float. Trust parameter
    """
    x, lbds = [x_init], [lbd]
    counter = 0
    while math.fabs(f(x[counter])) > err and counter < nMax: # iteration
        x_net = x[counter] - (f_diff(x[counter]))/(lbds[counter] + f_diff(x[counter])**2)*f(x[counter]) # update
        
        if f(x_net)**2 < f(x[counter])**2: # judge
            x.append(x_net)
            lbds.append(0.8 * lbds[counter])
        else:
            x.append(x[counter])
            lbds.append(lbds[counter] * 2)
        counter += 1
    return x, lbds, counter

def solve(u):
    x, lbds, cnt = lambert_marquardt(2, f(u), f_diff(u))
    y = [f(u)(i) for i in x]

    fig = plt.figure()
    plt.plot(range(1, cnt+2), y)
    plt.scatter(range(1,cnt+2), y, s=8)
    plt.xlabel('$k$')
    plt.ylabel('$f(x^{(k)})$')
    fig.savefig('18-1-1(u=2).pdf')

    fig = plt.figure()
    plt.scatter(range(1,cnt+2), lbds, s=8)
    plt.plot(range(1, cnt+2), lbds)
    plt.xlabel('$k$')
    plt.ylabel(r'$\lambda^{(k)}$')
    fig.savefig('18-1-2(u=2).pdf')
    print(x[-1], lbds[-1])
    plt.show()

def solve2():
    us = np.linspace(0,2,20)
    xx = []
    for u in us:
        x, lbds, cnt = lambert_marquardt(1, f(u), f_diff(u))
        xx.append(x[-1])

    fig = plt.figure()
    plt.scatter(us, xx, color='b', s=10, label='esmate point')
    x = np.linspace(0,1,20)
    y = [ i*math.exp(i) for i in x]
    plt.plot(y, x, color='r', label='real function')
    
    plt.xlabel('$u$')
    plt.ylabel('$x$')
    plt.legend()
    fig.savefig('18-1.pdf')
    plt.show()

def main():
    #solve(2)
    solve2()

if __name__ == '__main__':
    main()