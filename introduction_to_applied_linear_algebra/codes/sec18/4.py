#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.4
# Author: Utoppia
# Date  : 24 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
import math

def f(x, y):
    def wrapper(theta):
        return theta[0] * np.exp(theta[1]*x) - y
    return wrapper

def f_diff(x):
    def wrapper(theta):
        ans = np.zeros((len(x), len(theta)))
        for i in range(len(x)):
            ans[i] = [math.exp(theta[1]*x[i]), x[i]*theta[0]*math.exp(theta[1]*x[i])]
        return ans
    return wrapper


def lambert_marquardt(x_init, f, f_diff, lbd=1, err=10**(-6), nMax = 100):
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
    while norm(f(x[counter])) > err and counter < nMax: # iteration
        D = f_diff(x[counter])
        F = f(x[counter]).reshape(-1, 1)
        
        x_net = x[counter] - inv(D.T.dot(D) + lbds[counter]*np.identity(2)).dot(D.T.dot(F)).reshape(1,-1)[0] # update
        if norm(f(x_net)) < norm(f(x[counter])): # judge
            x.append(x_net)
            lbds.append(0.8 * lbds[counter])
        else:
            x.append(x[counter])
            lbds.append(lbds[counter] * 2)
            
        counter += 1
        print("iteration {}: {}, error: {}".format(counter, x[counter], norm(f(x[counter]))))
    return x, lbds, counter

def solve(x, y):
    thetas, lbds, cnt = lambert_marquardt([1,1], f(x, y), f_diff(x))
    theta = thetas[-1]

    fig = plt.figure()
    plt.scatter(x, y, color='g', marker='o', facecolors='none')

    x = np.linspace(-0.5, 5.5, 100)
    y = theta[0]*np.exp(theta[1]*x)
    plt.plot(x, y)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    fig.savefig('18-4.pdf')

    print(thetas[-1], lbds[-1])
    plt.show()

def main():
    x = np.linspace(0, 5, 6)
    y = np.array([5.2, 4.5, 2.7, 2.5, 2.1, 1.9])
    
    solve(x, y)

if __name__ == '__main__':
    main()