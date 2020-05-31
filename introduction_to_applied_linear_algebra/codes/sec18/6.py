#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.6
# Author: Utoppia
# Date  : 24 May 2020

import numpy as np 
from numpy.linalg import inv, pinv, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from utils import levenberg_marquadt

def sigmode(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def D_sigmode(x):
    1 - sigmode(x)**2

def residual(x, y, theta):
    ans = -y + theta[12]
    for i in range(0,3):
        r = theta[4*i + 1] * x[:,0] + theta[4*i + 2] * x[:,1] + theta[4*i + 3]
        phi = sigmode(r)
        ans = ans + theta[4*i]*phi
    return ans

def f(x, y, gamma):
    def wrapper(theta):
        upper = residual(x, y, theta)

        lower = np.sqrt(gamma) * theta 
        return np.hstack([upper, lower])
    return wrapper

def D_f(x, gamma):
    def wrapper(theta):
        c = [None, None, None]
        for i in range(0,3):
            r = theta[4*i + 1] * x[:,0] + theta[4*i + 2] * x[:,1] + theta[4*i + 3]
            phi = sigmode(r)
            D_phi = 1 - phi**2
            c[i] = np.vstack([phi, theta[4*i]*x[:,0]*D_phi, theta[4*i]*x[:,1]*D_phi, theta[4*i]*D_phi])
        
        upper = np.vstack([c[0], c[1], c[2], np.ones(len(x))]).T

        lower = np.sqrt(gamma)*np.eye(len(theta))

        return np.vstack([upper, lower])
    return wrapper

x = np.random.random((200, 2))
y = np.array([ x1*x2 for [x1, x2] in x[:,]])
thetas = {
    '0': np.zeros(13),
    '1': np.ones(13),
    #'0': np.random.random(13),
    #'1': np.random.random(13),
    'random': np.random.random(13)
}

gamma = 10 ** (-5)

DF = D_f(x, 10**(-5))
F = f(x, y, gamma)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for (key, theta0) in thetas.items():
    print('implement for theta0 = ', key)
    theta, history = levenberg_marquadt(theta0, F, DF)

    print(theta)

    print('rms', math.sqrt(mean_squared_error(y, residual(x,y,theta)+y)))
    print('objective', history['objectives'][-1])
    print('gradient', history['gradients'][-1])

    ax1.plot(history['objectives'][1:], label=r'$\theta_0 = {}$'.format(key))
    ax2.plot(history['gradients'][1:], label=r'$\theta_0 = {}$'.format(key))

ax1.set_xlabel('Iteration $k$')
ax1.set_ylabel('Objective value $f$')

ax2.set_xlabel('Iteration $k$')
ax2.set_ylabel('Norm of gradient of $f$')

ax1.legend()
ax2.legend()

ax1.set_ylim(-1, 15)
ax2.set_ylim(-0.2, 3)

#fig1.savefig('18-6-1.pdf')
#fig2.savefig('18-6-2.pdf')

# linear model fitting 
model = LinearRegression()
model.fit(x, y)
print('cofficients', model.intercept_, model.coef_)
print('rms', math.sqrt(mean_squared_error(y, model.predict(x))))

#plt.show()