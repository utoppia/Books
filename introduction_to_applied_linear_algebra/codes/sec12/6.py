#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 12.6
# Author: Utoppia
# Date  : 12 May 2020

import numpy as np 
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt


def latex_matrix_format(a):
    m, n = a.shape
    for i in range(m):
        print(' & '.join(map(lambda x: '{:6.3f}'.format(x), a[i,:])), '\\\\')

def main():
    c = np.array([1.0, 0.7, -0.3, -0.1, 0.05])
    C = np.array([
        [1.0  , 0    , 0    , 0    , 0    ],
        [0.7  , 1.0  , 0    , 0    , 0    ],
        [-0.3 , 0.7  , 1.0  , 0    , 0    ],
        [-0.1 , -0.3 , 0.7  , 1.0  , 0    ],
        [0.05 , -0.1 , -0.3 , 0.7  , 1.0  ],
        [0    , 0.05 , -0.1 , -0.3 , 0.7  ],
        [0    , 0    , 0.05 , -0.1 , -0.3 ],
        [0    , 0    , 0    , 0.05 , -0.1 ],
        [0    , 0    , 0    , 0    , 0.05 ]
    ])
    C_pseudo = pinv(C)
    print('Pseudo-inverse of C is')
    latex_matrix_format(C_pseudo)

    e_1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    h = C_pseudo.dot(e_1)
    print('h is ')
    print(h)

    # Plot
    plt.plot(c, label='$c$')
    plt.plot(h, label='$h$')
    plt.plot(C.dot(h), label=r"$h \ast c$")

    plt.grid(True)
    plt.legend()
    plt.xlabel('Time periods')
    plt.ylabel('Values')
    plt.title('Plot for exercise 12.6')
    plt.savefig('12-6.pdf')

if __name__ == '__main__':
    main()