#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 14.6
# Author: Utoppia
# Date  : 16 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error



def sub_f(x): # x: scalar
    if (x >= -0.5 and x < 0.1) or (x >= 0.5):
        return 1
    else :
        return -1

def classifier_sub(x):
    if x > 0:
        return 1
    else:
        return -1

def main():
    N = 200
    x = np.linspace(-1,1,N)
    f = np.frompyfunc(sub_f, 1, 1)
    classifier = np.frompyfunc(classifier_sub, 1, 1)
    y = f(x)

    model = [None for _ in range(9)]
    pre_y_p = [None for _ in range(9)]
    pre_y = [None for _ in range(9)]
    for i in range(9):
        model[i] = Pipeline([
            ('poly', PolynomialFeatures(degree=i)), # transform original feture to a polynomial feature list: x -> x, x^1, ...., x^d
            ('linear', LinearRegression(fit_intercept=False)) # use the linear model.
        ])
        model[i].fit(x.reshape(-1,1), y) # train the model

        pre_y_p[i] = model[i].predict(x.reshape(-1,1))
        pre_y[i] = classifier(pre_y_p[i])
        
        error_rate = np.sum(np.abs(y-pre_y[i])) / N
        print(i, error_rate)

    # -------------- Plot ---------------- # 

    plt.rcParams['xtick.labelsize'] = 'small'
    plt.rcParams['ytick.labelsize'] = 'small'
    plt.rcParams['legend.fontsize'] = 'x-small'
    
    
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8,8), tight_layout=True)

    for k in range(9):
        i, j = k//3, k%3
        ax[i, j].plot(x, pre_y_p[k], label=r'$\tilde{f}$') 
        ax[i, j].plot(x, pre_y[k], label=r'$\hat{f}$')
        ax[i, j].legend()

    fig.savefig('14-6.pdf')
    
if __name__ == '__main__':
    main()
