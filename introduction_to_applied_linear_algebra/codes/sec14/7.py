#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 14.7
# Author: Utoppia
# Date  : 17 May 2020

import numpy as np 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

def main():
    # Generate 200 examples with standard normal distribution
    N, p = 200, 2
    x = np.random.normal(size=(N,p))
    func = lambda x, y: 1 if x*y >= 0 else -1
    real_fun = np.frompyfunc(func, 2, 1)
    real_y = real_fun(x[:,0], x[:,1]).astype(np.int8)

    # Define the classifier 
    classifier = np.frompyfunc(lambda x: 1 if x>=0 else -1, 1, 1)

    # Define model
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)), # feature engineering
        ('linear', LinearRegression(fit_intercept=False)) # linear model
    ])

    model.fit(x, real_y) # modle fit on data
    tilde_y = model.predict(x) # prediction on original data
    pred_y = classifier(tilde_y).astype(np.int8) 

    print("The parameters are: ")
    print(model.named_steps['linear'].coef_)

    print('Confusion matrix is:')
    print(confusion_matrix(real_y, pred_y))

if __name__ == '__main__':
    main()