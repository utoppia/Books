#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 9.5
# Author: Utoppia
# Date  : 29 May 2020
import numpy as np 

class LinearDynamic:
    def __init__(self, A, C):
        self.A = A 
        self.C = C 
    def simulate(self, x0, T):
        states = np.zeros( (T, len(x0)) )
        outputs = []
        states[0] = x0 
        outputs.append(self.C @ states[0])

        for i in range(1, T):
            states[i] = self.A @ states[i-1]
            outputs.append(self.C @ states[i])
        return states, np.array(outputs, dtype=np.int)

# Fibonacci sequence 
A = np.array([
    [1, 1],
    [1, 0]
])
C = np.array([1, 0])
x0 = np.array([1,0])

fibo = LinearDynamic(A, C)

_, outputs = fibo.simulate(x0, 20)

# modified Fibonacci sequence 
A = np.array([
    [1, -1],
    [1, 0]
])

modi_fibo = LinearDynamic(A, C)
_, output2 = modi_fibo.simulate(x0, 20)

print(outputs)
print(output2)