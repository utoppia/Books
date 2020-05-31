#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Section 19 Util Functions
# Author: Utoppia
# Date  : 31 May 2020

from numpy.linalg import norm, inv, pinv
from numpy import eye, sqrt, vstack, hstack

def levenberg_marquadt(x0, F, DF, lamda0=1, nMax=100, tol=10**(-6)):
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

def penalty(x0, F, DF, G, DG, mu0=1, nMax=100, tol=10**(-6)):
    '''
    Panalty algorithm to solve the constrained nonlinear least squares problem :
        minimize    || F(x) ||^2 
        subject to  G(x) = 0
    ------------------------------------------
    Parameters: 
        x0: initial feature vector 
        F: function 
        DF: derivative function of F 
        G: constrained function 
        DG: derivative function of G
    '''

    mu, x = mu0, x0
    status = 0
    u = []
    for k in range(0, nMax):
        f = lambda x: hstack([ F(x), sqrt(mu) * G(x) ])
        Df = lambda x: vstack([ DF(x), sqrt(mu) * DG(x) ])
        x, _ = levenberg_marquadt(x, f, Df)
        u.append(mu)
        if norm(G(x)) < tol: 
            status = 1
            break
        mu = 2 * mu
    return x, mu, {'penalty': u}

def augmented_lagrangian(x0, z0, F, DF, G, DG, mu0=1, nMax=100, tol=10**(-6), ttol=10**(-6)):
    '''
    Augmented Lagrangian algorithm to solve the constrained nonlinear least squares problem :
        minimize    || F(x) ||^2 
        subject to  G(x) = 0
    changed to 
        minimize || F(x) ||^2 + mu * || G(x) + z / (2mu)||^2
    with update of z:
        z = z + 2 * u * G(x)

    ------------------------------------------
    Parameters: 
        x0: initial feature vector 
        z0: Optimal Lagrange multipliers
        F: function 
        DF: derivative function of F 
        G: constrained function 
        DG: derivative function of G
    '''
    mu, x, z = mu0, x0, z0 
    status = 0
    residual_y = [norm(F(x))**2]
    residual_g = [norm(G(x))**2]
    us = []
    for k in range(nMax):
        f = lambda x: hstack([ F(x), sqrt(mu) * (G(x)+(z/(2*mu))) ])
        Df = lambda x: vstack([ DF(x), sqrt(mu) * DG(x) ])
        xk, _ = levenberg_marquadt(x, f, Df)
        residual_y.append(norm(F(xk))**2)
        residual_g.append(norm(G(xk))**2)
        us.append(mu)
        if norm(G(xk)) < tol: # if terminal condition 1 holds
            x = xk
            status = 1
            break

        if norm(2*DF(xk).T @ F(xk) + DG(xk).T @ z) < ttol: # if terminal condition 2 holds
            x = xk
            status = 2
            break

        z = z + 2 * mu * G(xk) # Update z
        if norm(G(xk)) < 0.25 * norm(G(x)): # Update mu
            mu = mu 
        else :
            mu = 2 * mu
        x = xk 
        
    return x, z, status, {'residual1': residual_y, 'residual2': residual_g, 'penalty': us}