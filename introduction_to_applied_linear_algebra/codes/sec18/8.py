#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.7
# Author: Utoppia
# Date  : 25 May 2020

import numpy as np 
from numpy import cos, sin, diag, ones, zeros, pi
from utils import levenberg_marquadt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def get_points(t, theta):
    c1, c2, r, delta, alpha = theta
    x = c1 + r * cos(alpha + t) + delta * cos(alpha - t)
    y = c2 + r * sin(alpha + t) + delta * sin(alpha - t) 
    return np.vstack([x, y]).T

def f_ (points):
    def wrapper(x):
        t, theta = x[:-5], x[-5:]
        c1, c2, r, delta, alpha = theta
        upper = c1 + r * cos(alpha + t) + delta * cos(alpha - t) - points[:,0]
        lower = c2 + r * sin(alpha + t) + delta * sin(alpha - t) - points[:,1]
        return np.hstack([upper, lower])
    return wrapper

def dFt(t, theta):
    c1, c2, r, delta, alpha = theta
    b = -r * sin(alpha + t) + delta * sin(alpha-t)
    return diag(b)
def dGt(t, theta):
    c1, c2, r, delta, alpha = theta
    d = r * cos(alpha + t) - delta * cos(alpha-t)
    return diag(d)
def dFtheta(t, theta):
    n = len(t)
    c1, c2, r, delta, alpha = theta
    return np.vstack([
        ones(n), zeros(n), cos(alpha+t), cos(alpha-t), -r*sin(alpha+t)-delta*sin(alpha-t)
    ]).T
def dGtheta(t, theta):
    n = len(t)
    c1, c2, r, delta, alpha = theta
    return np.vstack([
        zeros(n), ones(n), sin(alpha+t), sin(alpha-t), r*cos(alpha+t)+delta*cos(alpha-t)
    ]).T

def D_f_ ():
    def wrapper(x):
        t, theta = x[:-5], x[-5:]
        c1, c2, r, delta, alpha = theta
        DFt = dFt(t, theta)
        DFtheta = dFtheta(t, theta)
        DGt = dGt(t, theta)
        DGtheta = dGtheta(t, theta)

        return np.block([
            [DFt, DFtheta],
            [DGt, DGtheta]
        ])
    return wrapper

def f_min (points, theta):
    def wrapper(t):
        c1, c2, r, delta, alpha = theta
        upper = c1 + r * cos(alpha + t) + delta * cos(alpha - t) - points[:,0]
        lower = c2 + r * sin(alpha + t) + delta * sin(alpha - t) - points[:,1]
        return np.hstack([upper, lower])
    return wrapper
def D_f_min (theta):
    def wrapper(t):
        DFt = dFt(t, theta)
        DGt = dGt(t, theta)

        return np.vstack([DFt, DGt])
    return wrapper

def plot(points, t, theta, filename):
    fig, ax = plt.subplots()
    ps = get_points(t, theta)
    
    for (p1, p2) in zip(points, ps):
        ax.plot([p1[0], p2[0]], [p1[1],p2[1]], color='gray')

    ax.scatter(points[:,0], points[:,1], s=10, color='green', facecolors='none', marker='o')
    ax.scatter(ps[:,0], ps[:,1], s=10, color='red', marker='o')

    t = np.linspace(0, 2*pi, 100)
    ps = get_points(t, theta)
    ax.plot(ps[:,0], ps[:,1], color='b')
    fig.savefig(filename)
    plt.show()

points = np.array([
    (0.5, 1.5), (-0.3, 0.6), (1.0, 1.8), (-0.4, 0.2), (0.2, 1.3),
    (0.7, 0.1), (2.3, 0.8), (1.4, 0.5), (0.0, 0.2), (2.4, 1.7)
])

def solve(points):
    c = np.average(points, axis=0)

    theta0 = [c[0], c[1], 1, 0, 0]
    df = D_f_min(theta0)
    f = f_min(points, theta0)
    t0 = zeros(len(points))

    t, history = levenberg_marquadt(t0, f, df)
    print(t) 
    print(history['objectives'][-1])
    plot(points, t, theta0, '18-8-1.pdf')

    # ----------------------- #
    df = D_f_()
    f = f_(points)
    x0 = np.hstack([t, theta0])
    x, history = levenberg_marquadt(x0, f, df)
    print(history['objectives'][-1])
    print(x)
    t, theta = x[:-5], x[-5:]
    plot(points, t, theta, '18-8-2.pdf')

def create_question_figure():
    t = 2*pi*np.random.random(50)
    theta_real = np.array([1, 1, 1, 0.5, 30*pi/180])
    points = get_points(t, theta_real)
    points = points + np.random.random((50, 2))*0.1

    solve(points)

solve(points)