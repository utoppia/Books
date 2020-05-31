#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 13.3
# Author: Utoppia
# Date  : 14 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def log_plot(x, y, y1):

    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True 
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.scatter(x, y, color='tab:blue', marker='o', facecolors='none')
    plt.plot(x,y1, color='r')
    
    plt.yscale('log')
    
    plt.xlabel('Year')
    plt.ylabel('Transistors')
    plt.xlim(right=2005)
    plt.savefig('13-3.pdf')
    #plt.show()

def main():
    t = np.array([1971, 1972, 1974, 1978, 1982, 1985, 1989, 1993, 1997, 1999, 2000, 2002, 2003])
    N = np.array([2250, 2500, 5000, 29000, 120000, 275000, 1180000, 3100000, 7500000, 24000000, 42000000, 220000000,410000000])

    # log_plot(t, N)

    p = len(t) # sizes of data set
    x = t - 1970
    y = np.log10(N)

    A = np.block([ [np.ones(p)], [x]]).T 
    theta = pinv(A).dot(y)
    print('The coefficients are: ')
    print(theta)
    predict_y = theta[0] + theta[1]*x 
    predict_N = np.power(10, predict_y)
    print('RMS is', norm(y - predict_y) / np.sqrt(p)) 
    print('prediction y in 2015 is', theta.dot(np.array([1, 2015-1970])))

    # use skcit-learn module to re-calculate this model
    print('\n'+30*"-"+'\n\tUse skcit-learn module\n'+30*'-')
    reg = linear_model.LinearRegression()
    reg.fit(x.reshape(-1,1) , y)
    print('The coefficients are: ')
    print(reg.intercept_, reg.coef_)
    predict_y = reg.predict(x.reshape(-1,1))
    print('RMS between y and predict_y is', np.sqrt(mean_squared_error(y, predict_y)))

    print('prediction y in 2015 is ', reg.predict( [ [2015-1970] ] )) 

    log_plot(t, N, predict_N)
if __name__ == '__main__':
    main()