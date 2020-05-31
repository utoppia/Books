#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 7.11
# Author: Utoppia
# Date  : 06 May 2020

import numpy as np 
import matplotlib.pyplot as plt
import time

class Single:
    def __init__(self, m, impluse, equalizer):
        self.impluse = impluse
        self.equalizer = equalizer
        np.random.seed(int(time.time()))
        self.single = np.random.choice([1,-1], m)
        self.recieved = np.convolve(self.impluse, self.single)

    def plot(self):
        self.fig = plt.figure()
        
        plt.plot(self.single, label='orignal single')
        plt.plot(self.recieved, label='recieved single')
        plt.plot(np.convolve(self.recieved, self.equalizer), label='equalized single')

        plt.xlim((0, 70))
        plt.xlabel('Time series')
        plt.ylabel('Single values')
        plt.title('Example for Exercise 7.11')
        plt.legend(fontsize='small')
        plt.savefig('11.png')

def main():
    impluse = np.array([1, 0.7, -0.3])
    equalizer = np.array([0.9, -0.5, 0.5, -0.4, 0.3, -0.3, 0.2, -0.1])
    m = 50
    single = Single(m, impluse, equalizer)
    single.plot()

main()