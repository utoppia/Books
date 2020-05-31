#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.2
# Author: Utoppia
# Date  : 24 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
import math

def f(c):
    def N(r):
        ans = 0
        p = 1
        for i in range(0, len(c)):
            ans += c[i] / p
            p = p * (1+r)
        return ans 
    return N

def f_diff(c):
    def N_diff(r):
        ans = 0
        p = (1+r)
        for i in range(0, len(c)):
            ans += i*c[i]/p
            p = p * (1+r)
        return -ans 
    return N_diff

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

def solve(c):
    x, lbds, cnt = lambert_marquardt(0, f(c), f_diff(c))
    y = [f(c)(i)**2 for i in x]

    fig = plt.figure()
    plt.plot(range(1, cnt+2), y)
    plt.scatter(range(1,cnt+2), y, s=8)
    plt.xlabel('$k$')
    plt.ylabel('$N(r^{(k)})^2$')
    fig.savefig('18-2.pdf')

    print(x[-1], lbds[-1])
    plt.show()



def main():
    c = [-1, -1, -1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    #print(f_diff(c)(0))
    solve(c)

if __name__ == '__main__':
    main()