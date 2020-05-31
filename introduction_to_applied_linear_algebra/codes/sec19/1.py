#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exercise 19.1
# Author: Utoppia
# Date  : 31 May 2020

from numpy import array, linspace
from numpy.linalg import norm
from utils import augmented_lagrangian, penalty
import matplotlib.pyplot as plt

f = lambda x: array([(x[0]-1)**2 + (x[1]-1)**2 + (x[2]-1)**2 ])
Df = lambda x: array([
    [2*x[0]-2, 2*x[1]-2, 2*x[2]-2]
])

g = lambda x: array([ x[0]**2 + 0.5*x[1]**2 + x[2]**2-1, 0.8*x[0]**2 + 2.5*x[1]**2 + x[2]**2 + 2*x[0]*x[2] - x[0]-x[1]-x[2]-1 ])
Dg = lambda x: array([
    [2*x[0], x[1], 2*x[2]],
    [1.6*x[0] + 2*x[2]-1, 5*x[1]-1, 2*x[2]+2*x[0]-1]
])

x0 = array([0,0,0])
z0 = array([0,0])

x, z, status, history = augmented_lagrangian(x0, z0, f, Df, g, Dg, tol=10**(-5), ttol=10**(-5))
iterator = len(history['residual1'])
print('Using Augmented Lagrangian algorithm')
print('x is', x)
print('minimum objective is', norm(f(x))**2 )
print('Iterator is', iterator )


# Plot 
fig, ax = plt.subplots()
k = linspace(1, iterator+1, iterator)
ax.plot(k, history['residual1'], label='$||f(x)||^2$')
ax.plot(k, history['residual2'], label='$||g(x)||^2$')
ax.set_xlim(1)
ax.set_xlabel('Iterator $k$')
ax.set_ylabel('Residuals')
ax.legend()
fig.savefig('19-1-1.pdf')

fig, ax = plt.subplots()
ax.plot(k[:-1], history['penalty'], label='Augmented Lagrangian')
ax.set_xlabel('Iterator $k$')
ax.set_ylabel('Penalty parameter')
ax.set_xlim(left=1)
fig.savefig('19-1-2.pdf')

x1, _, history = penalty(x0, f, Df, g, Dg, tol=10**(-5))
iterator = len(history['penalty'])
print(50*'-')
print('Using penalty method')
print('x is', x1)
print('Minimum objectice is', norm(f(x))**2)
print('Iterator is', iterator + 1)

k = linspace(1, iterator+1, iterator)
ax.plot(k, history['penalty'], label='Penalty')
ax.set_xlim(right=iterator+1)
ax.legend()
fig.savefig('19-1-3.pdf')
#plt.show()