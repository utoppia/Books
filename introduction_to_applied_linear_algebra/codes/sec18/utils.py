#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 18.6
# Author: Utoppia
# Date  : 24 May 2020

from numpy.linalg import norm, inv
from numpy import eye

def levenberg_marquadt(x0, F, DF, lamda0=1, nMax=100, tol=10**(-8)):
    """
    Levenberg-Marquadt algorithm to solve nonlinear least squres problem with 
    objective F(x) and derivative matrix DF(x).
         minimize ||F(x)||^2 + `\lambda'*||x-x_k||^2 
    --------------------------------------------
    Parameters:
        x0: initialization of x, array-like 
        F: objevtice function, callable
        DF: derivative matrix function, callable
        lamda0: float
        tol: if gradient of f smaller than tol, quit the iteration
    Return:
        x: array like
        history: dict object
    """
    n = len(x0)
    x, lamda = x0, lamda0

    objectives = [0]
    residuals = [0]
    gradients = [0]

    for k in range(nMax):

        fk = F(x)
        Dfk = DF(x)
        objectives.append(norm(fk)**2)
        gradients.append(norm(2*Dfk.T @ fk))

        if norm( 2 * Dfk.T @ fk) < tol:
            break

        xk = x - inv( Dfk.T @ Dfk + lamda * eye(n) ) @ Dfk.T @ fk

        if norm(F(xk)) < norm(fk):
            x = xk
            lamda = 0.8 * lamda
        else:
            lamda = 2 * lamda

    return x, { 'objectives': objectives, 'gradients': gradients }