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

def force(a1, a2, L1, L2, k, m, g=9.8):
    def F(x, a, L, k):
        dis = norm(a-x)
        return k * (dis - L)/(L*dis)*(a-x)
    def wrapper(x):
        return -m*g*np.array([0, 1]) + F(x,a1,L1,k) + F(x, a2, L2, k)     
    return wrapper

def force_diff(a1, a2, L1, L2, k, m, g=9.8):
    def F_diff(x, a, L, k):
        dis = norm(a-x)
        diff = np.array([
            [1/L-1/dis+(a[0]-x[0])**2/dis**3, (a[0]-x[0])*(a[1]-x[1])/dis**3],
            [(a[0]-x[0])*(a[1]-x[1])/dis**3, 1/L-1/dis+(a[1]-x[1])**2/dis**3]
        ])
        return -k * diff
    def wrapper(x):
        return F_diff(x, a1, L1, k) + F_diff(x, a2, L2, k)
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
    x, lbds = np.array([x_init]), [lbd]
    counter = 0
    while norm(f(x[counter])) > err and counter < nMax: # iteration
        D = f_diff(x[counter])
        F = f(x[counter]).reshape(-1, 1)
        
        x_net = x[counter] - inv(D.T.dot(D) + lbds[counter]*np.identity(2)).dot(D.T.dot(F)).reshape(1,-1)[0] # update
        if norm(f(x_net)) < norm(f(x[counter])): # judge
            x = np.vstack([x, x_net])
            lbds.append(0.8 * lbds[counter])
        else:
            x = np.vstack([x, x[counter]])
            lbds.append(lbds[counter] * 2)
            
        counter += 1
        print("iteration {}: {}, error: {}, lambda: {}".format(counter, x[counter], norm(f(x[counter])), lbds[counter]))
    return x, lbds, counter

def solve(a1, a2, L1, L2, k, m):

    f = force(a1, a2, L1, L2, k, m)
    f_diff = force_diff(a1, a2, L1, L2, k, m)
    x0 = np.array([0, 0])
    xs, lbds, cnt = lambert_marquardt(x0, f, f_diff)
    x = xs[-1]

    fig = plt.figure()

    k = range(1, cnt+2)
    plt.scatter(k, xs[:,0], s=10)
    plt.plot(k, xs[:,0])
    plt.xlabel('$k$')
    plt.ylabel('$x$')
    #fig.savefig('18-5-1.pdf')

    fig = plt.figure()
    plt.scatter(k, xs[:,1], s=10)
    plt.plot(k, xs[:,1])
    plt.xlabel('$k$')
    plt.ylabel('$y$')
    #fig.savefig('18-5-2.pdf')

    fig = plt.figure()
    residual = [norm(f(x)) for x in xs]
    plt.scatter(k, residual, s=10)
    plt.plot(k, residual)
    plt.xlabel('$k$')
    plt.ylabel('$||f(x)||$')
    #fig.savefig('18-5-3.pdf')

    x = np.linspace(-1.5, 3.5, 100)
    y = np.linspace(-1.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    print(X.shape)
    Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            Z[i,j] = norm(f(np.array([X[i,j], Y[i,j]])))
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    plt.scatter(a1[0], a1[1], marker='o', color='black')
    plt.scatter(a2[0], a2[1], marker='o', color='black')
    plt.scatter(xs[:,0], xs[:,1], marker='*', color='r')
    plt.plot(xs[:,0], xs[:,1], color='r')
    plt.arrow(xs[0,0], xs[0,1], (xs[1,0]-xs[0,0])/2, (xs[1,1]-xs[0,1])/2, head_width=0.03, color='r')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    fig.savefig('18-5-4.pdf')
    plt.show()

def main():
    a1 = np.array([3,2])
    a2 = np.array([-1,1])
    L1, L2 = 3, 2
    m, k = 1, 100
    
    solve(a1, a2, L1, L2, k, m)

if __name__ == '__main__':
    main()