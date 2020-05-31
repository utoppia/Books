#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.7
# Author: Utoppia
# Date  : 25 May 2020

import numpy as np 
from math import cos, sin
from utils import levenberg_marquadt

def f(p_des, l):
    def wrapper(theta):
        return np.array([
            l[0]*cos(theta[0]) + l[1]*cos(theta[0]+theta[1]) - p_des[0],
            l[0]*sin(theta[0]) + l[1]*sin(theta[0]+theta[1]) - p_des[1]
        ])
    return wrapper

def Df(p_des, l):
    def wrapper(theta):
        return np.array([
            [-l[0]*sin(theta[0])-l[1]*sin(theta[0]+theta[1]), -l[1]*sin(theta[0]+theta[1])],
            [l[0]*cos(theta[0])+l[1]*cos(theta[0]+theta[1]), l[1]*cos(theta[0]+theta[1])]
        ])
    return wrapper

p_des_list = np.array([
    [1.0, 0.5],
    [-2.0, 1.0],
    [-0.2, 3.1]
])
l = [2, 1]

for p_des in p_des_list:
    F = f(p_des, l)
    DF = Df(p_des, l)

    theta, history = levenberg_marquadt([0,0], F, DF)
    print(theta*180/np.pi)