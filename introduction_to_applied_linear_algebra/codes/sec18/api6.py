#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.4
# Author: Utoppia
# Date  : 24 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def gradient(x, theta):
    """
    Grandient of function with respect theta 
    -------------
    Parameters:
        x: 2-vector array-like
        theta: array-like
    """
    r = [0,0,0]
    for i in range(0,3):
        r[i] = theta[4*i + 1]*x[0] + theta[4*i + 2]*x[1] + theta[4*i+3]
    phi = [sigmoid(u) for u in r]
    phi_d = [1-u**2 for u in phi]

    return np.array([
        phi[0], theta[0]*x[0]*phi_d[0], theta[0]*x[1]*phi_d[0], theta[0]*phi_d[0], 
        phi[1], theta[4]*x[0]*phi_d[1], theta[4]*x[1]*phi_d[1], theta[4]*phi_d[1],
        phi[2], theta[8]*x[0]*phi_d[2], theta[8]*x[1]*phi_d[2], theta[8]*phi_d[2],
        1
        ])

def model_f(x, theta):
    """
    Value of the function  
    -------------
    Parameters:
        x: 2-vector array-like
        theta: array-like
    """
    r = [0,0,0]
    for i in range(0,3):
        r[i] = theta[4*i + 1]*x[0] + theta[4*i + 2]*x[1] + theta[4*i+3]
    phi = [sigmoid(u) for u in r]
    return theta[0]*phi[0] + theta[4]*phi[1] + theta[8]*phi[2] + theta[12]

def residual(x, y):
    """
    Residual vector of model f(x;theta) and real value y
    --------------
    Parameters:
        x: matrix with dimension (m,2)
        y: array like
    Return:
        (m,1) matrix
    """
    def wrapper(theta):
        ans = np.zeros((len(x), 1))
        for i in range(len(x)):
            ans[i] = model_f(x[i], theta) - y[i]
        return ans
    return wrapper

def jocobi_diff(x, y):
    """
    Derivation matrix 
    -----------------
    Parameter:
        x: matrix with dimension (m,2)
        y: scalar value
    Return:
        func: calable
    """
    def wrapper(theta):
        jocobi = np.zeros((len(x), len(theta)))
        for i in range(len(x)):
            jocobi[i] = gradient(x[i], theta)
        return jocobi
    return wrapper

def lambert_marquardt(x0, f, f_diff, obj, gradient, lbd=1, gamma=10*(-5), err=10**(-6), nMax = 100):
    """
    Lambert-Marquardt algorithm to solve nonlinear least squares problem,
    for 1 dimension variable
    -------------------
    Parameters:
        x0 : Initialization, array like, 13-vector
        f : residual function
        f_diff: calable function, to get derivation matrix of f
        obj: calable function, objective
        lbd: Float. Trust parameter
    """
    x, lbds = np.array([x0]), [lbd]
    
    counter = 0
    change = 100
    objs = np.array(obj(x[counter]))
    gras = np.array(gradient(x[counter]))
    while change > err and counter < nMax: # iteration
        print('\rInteration: {}'.format(counter+1), end='', flush=True)
        D = f_diff(x[counter])
        F = f(x[counter])
        
        tmp =D.T.dot(D)
        A = tmp + lbds[counter]*np.identity(13) # A = D^TD + lambda I
        B = A + gamma*np.identity(13) # B = D^TD + lambda I + gamma I
        B = inv(B) # B = B^(-1)
        A = B.dot(A) # A = B^(-1) A
        c = D.T.dot(F) # c = D^T . F
        x_net = A.dot(x[counter].reshape(-1,1)) - B.dot(c)
        x_net = x_net.reshape(1,-1)[0]
        change = obj(x[counter])-obj(x_net)
        if change >= 0: # judge
            x = np.vstack([x, x_net])
            lbds.append(0.8 * lbds[counter])
        else:
            x = np.vstack([x, x[counter]])
            lbds.append(lbds[counter] * 2)
            change = 100
        counter += 1
        objs = np.vstack([objs, obj(x[counter])])
        gras = np.vstack([gras, gradient(x[counter])])
        
        #print("iteration {}: {}, error: {}, lambda: {}".format(counter, x[counter], obj(x[counter]), lbds[counter]))
    return x, lbds, counter, objs, gras

def linear_model_fit(x, y):
    model = LinearRegression()
    model.fit(x, y)
    print(model.coef_, model.intercept_)
    print(np.sqrt(mean_squared_error(y, model.predict(x))))

def solve(x, y):

    fig = plt.figure()

    theta_inits = np.array([
        np.zeros(13),
        np.ones(13),
        np.random.random(13)
    ])

    f = residual(x, y)
    f_diff = jocobi_diff(x, y)
    gamma = 0 #10**(-5)
    obj = lambda x: norm(f(x))**2 + gamma*norm(x)**2
    gradient = lambda x: 2 * norm(f_diff(x).T.dot(f(x)) + gamma*x.reshape(-1,1))
    
    for i in range(3):
        theta0 = theta_inits[i]
        print("-"*50)
        print("\t Initialization of theta = ")
        print(theta0)
        
        thetas, lbds, cnt, objs, gras = lambert_marquardt(theta0, f, f_diff, obj, gradient, gamma=gamma)

        print("\nOver.")
        print("-> Paramaters are:")
        print("\t", thetas[-1])
        print("-> RMS of model")
        print("\t", np.sqrt(mean_squared_error(y, f(thetas[-1]).reshape(1,-1)[0]+y)))
        print('-> Objective is')
        print("\t", objs[-1])
        print('-> gradient of objective is')
        print("\t", gras[-1])
        print("\n")

        k = range(1, cnt+2)

        plt.scatter(k, gras, s=10)
        plt.plot(k, gras, label=r'${}$th $\theta_0$'.format(i+1))

        

        #plt.scatter(x, y, facecolors='none', color='green', marker='o')

    plt.legend()
    plt.show()

def main():
    x = np.random.random((200,2))
    y = np.array([ x1*x2 for [x1,x2] in x ])

    solve(x, y)
    linear_model_fit(x,y)


def api_solve(x, y, theta0, gamma):
    f = residual(x, y)
    f_diff = jocobi_diff(x, y)
    obj = lambda x: norm(f(x))**2 + gamma*norm(x)**2
    gradient = lambda x: 2 * norm(f_diff(x).T.dot(f(x)) + gamma*x.reshape(-1,1))
    
    print("\t Initialization of theta = ")
    print(theta0)
    
    thetas, lbds, cnt, objs, gras = lambert_marquardt(theta0, f, f_diff, obj, gradient, gamma=gamma)

    print("\nOver.")
    print("-> Paramaters are:")
    print("\t", thetas[-1])
    print("-> RMS of model")
    print("\t", np.sqrt(mean_squared_error(y, f(thetas[-1]).reshape(1,-1)[0]+y)))
    print('-> Objective is')
    print("\t", objs[-1])
    print('-> gradient of objective is')
    print("\t", gras[-1])
    print("\n")
    return thetas[-1]