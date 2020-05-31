# -*- coding:utf-8 -*-
# Python 3.6+
# Tool Functions
# Author: Utoppia
# Last Update: 12 May 2020

def latex_matrix_format(a):
    m, n = a.shape
    for i in range(m):
        print(' & '.join(map(lambda x: str(x)[:6], a[i,:])), '\\\\')