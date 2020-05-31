#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 10.30
# Author: Utoppia
# Date  : 08 May 2020

import numpy as np

def gram_schmidit(a):
    '''    
    This function use Gram-Schmidt algorithm to check whether list a1, ..., ak are linear independent.
        
    Parameters:
    ----------
    a: vector list
        A list of vectors

    Returns:
    --------
        Boolean, Vector list
    '''

    size = len(a)
    q = np.zeros(a.shape)
    for i in range(size):
        # Orthogonalization
        q[i] = a[i]
        for j in range(i):
            q[i] = q[i] - (q[j].dot(a[i]))*q[j]

        # Test for linear dependence. if q[i] = 0, quit
        if not np.any(q[i]):
            return False, None 
        
        # Normalization 
        q[i] = q[i] / np.linalg.norm(q[i])
    
    return True, q

def QR_factorization(A):
    '''

    Parameters:
    ----------
    A: Numpy Array Object
        Denote a matrix 
    '''

    n, k = A.shape 
    Q = np.zeros((n, k))
    R = np.zeros((k, k))
    for i in range(k): # There are k n-column-vectors.
        # Orthogonalization
        Q[:,i] = A[:,i]
        for j in range(i):
            R[j,i] = A[:,i].dot(Q[:,j])
            Q[:,i] = Q[:,i] - R[j,i]*Q[:,j]
        
        # Test for linear dependence. if Q[:,i] = 0, quit
        if not np.any(Q[:,i]):
            return False, None, None
        
        # Normalization
        R[i,i] = np.linalg.norm(Q[:,i])
        Q[:,i] = Q[:,i] / R[i,i]

    return True, Q, R

if __name__ == '__main__':

    # Test for this function
    a = np.array([
        [-3, -4],
        [4,   6],
        [1,   1]
    ])

    print(gram_schmidit(np.transpose(a)))
    print(QR_factorization(a))
    print(np.linalg.qr(a))