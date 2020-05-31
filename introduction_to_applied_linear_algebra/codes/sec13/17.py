#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 13.17
# Author: Utoppia
# Date  : 14 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['legend.fontsize'] = 'x-small'
def f(x):
    return (1+x)/(1+5*(x**2))

def main():
    x = np.linspace(-1,1,11)
    y = f(x)

    model = [None for _ in range(9)]
    for i in range(9):
        model[i] = Pipeline([
            ('poly', PolynomialFeatures(degree=i)), # transform original feture to a polynomial feature list: x -> x, x^1, ...., x^d
            ('linear', LinearRegression(fit_intercept=False)) # use the linear model.
        ])
        model[i].fit(x.reshape(-1,1), y) # train the model

    # Validate the model by test data set 
    # Generate test data set 
    u = np.linspace(-1.1, 1.1, 10)
    y_test = f(u)

    test_RMS = np.zeros(9) # init
    train_RMS = np.zeros(9) # init

    for i in range(9):
        #print(mean_squared_error(y, model[i].predict(x.reshape(-1,1))))
        train_RMS[i] = np.sqrt(mean_squared_error(y, model[i].predict(x.reshape(-1,1))))
        test_RMS[i] = np.sqrt(mean_squared_error(y_test, model[i].predict(u.reshape(-1,1))))

    print('-'*50)
    print('{:<10s}{:<15s}{:<15s}'.format('Degree', 'Train RMS', 'TEST RMS'))
    print('-'*50)
    for i in range(9):
        print('{:<10d}{:<15.3f}{:<15.3f}'.format(i, train_RMS[i], test_RMS[i]))
    print('-'*50)


    # ------------ Plot the prediction of models --------------# 
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8,8), tight_layout=True)
    new_x = np.linspace(-1.1,1.1,100)
    new_y = f(new_x)

    for i in range(9):
        ax[i//3, i%3].scatter(x,y)
        ax[i//3, i%3].plot(new_x, new_y, label='real')
        ax[i//3, i%3].plot(new_x, model[i].predict(new_x.reshape(-1,1)), label="degree {}".format(i))
        
        ax[i//3, i%3].legend(loc='lower center')
        ax[i//3, i%3].spines['top'].set_visible(False)
        ax[i//3, i%3].spines['right'].set_visible(False)
        ax[i//3, i%3].annotate(r'$x$', xy=(1.05, 0.03), ha='left', va='top', xycoords='axes fraction', fontsize='small')
        ax[i//3, i%3].annotate(r'$\hat{f}$', xy=(0.1, 1.1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', size='small')

        ax[i//3, i%3].set_xlim(-1.1,1.1)
    
    fig.savefig('13-17.pdf')
    

    # ----------- Plot the Train RMS and Test RMS ----------- #
    fig = plt.figure()
    plt.plot(train_RMS, label='train RMS error')
    plt.plot(test_RMS, label='test RMS error')
    plt.xlim(0, 8)
    plt.xlabel('Order Degree of the Polynomial Model')
    plt.ylabel('Root Mean Square')
    plt.title('RMS of different models')
    plt.legend()
    fig.savefig('13-17-2.pdf')

    plt.show()
if __name__ == '__main__':
    main()
