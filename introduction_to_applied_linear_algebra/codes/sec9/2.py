#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 9.2
# Author: Utoppia
# Date  : 06 May 2020

import numpy as np 
import matplotlib.pyplot as plt
import time

class Economy:
    def __init__(self, B, a):
        self.a = a
        self.B = B

    def simulation(self, T):
        prediction = np.zeros((T, len(self.a)))
        prediction[0] = self.a
       
        for i in range(1, T):
            prediction[i] = self.B.dot(prediction[i-1])
        
        return prediction

def plot_sectors(T, prediction):
    fig = plt.figure()

    x = np.linspace(1, 20, 20)
    (m, n) = prediction.shape
    for i in range(n):
        plt.plot(x, prediction[:, i], label='sector - {}'.format(i))

    plt.xlabel('Time (year)')
    plt.ylabel('output (billions of dollars)')
    plt.title('Output of sectors')
    plt.xticks(np.linspace(0,20,11))
    plt.xlim(1, 20)
    plt.grid(True)
    plt.legend()
    
    fig.savefig('9-2-1.pdf')

def plot_total_output(T, prediction):
    fig = plt.figure()
    x = np.linspace(0,20,20)
    (m,n) = prediction.shape 
    total = [item.sum() for item in prediction]
    plt.plot(x, total)

    plt.xlabel('Time (year)')
    plt.ylabel('output (billions of dollars)')
    plt.title('Total Economic Output')
    plt.xticks(np.linspace(0,20,11))
    plt.xlim(1, 20)
    plt.grid(True)
    
    plt.savefig('9-2-2.pdf')

def main():
    B = np.array([
        [0.10, 0.06, 0.05, 0.70],
        [0.48, 0.44, 0.10, 0.04],
        [0.00, 0.55, 0.52, 0.04],
        [0.04, 0.01, 0.42, 0.52]
    ])
    a_init = np.array([0.6, 0.9, 1.3, 0.5])
    T = 20

    economy = Economy(B, a_init)
    prediction = economy.simulation(T)
    plot_sectors(T, prediction)
    plot_total_output(T, prediction)

main()