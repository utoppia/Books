#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 17.6
# Author: Utoppia
# Date  : 26 May 2020

import numpy as np 
from numpy import array, eye, zeros, linspace, sqrt
from numpy.linalg import pinv
import matplotlib.pyplot as plt 

def least_square_constrained(A, b, C, d):
    """
    Solve the linear constrained least squares problem:
        minimize    ||Ax - b||^2
        subject to  Cx = d
    let x' = (x, z), z is the Langrange multipliers of constraits equalities, 
    problem above is same to solve 
        | 2A^TA C^T | | x |   | 2A^Tb |
        |           | |   | = |       |
        | C     0   | | z |   | d     |
    ---------------------------------------
    Parameters:
    A : matrix 
    b : array-like
    C : matrix 
    d : array-like
    """
    _, n = A.shape
    p, _ = C.shape 
   
    M = np.block([
        [2 * A.T @ A, C.T],
        [C, zeros((p,p))]
    ])
    N = np.hstack([2 * A.T @ b, d])

    X = pinv(M) @ N
    x, z = X[:n], X[n:]
    return x, z

def simulation_open_loop(A, x0, T):
    '''
    Simulation for a linear dynamical system without control, ie, 
    open-loop, in time series t = 1, ..., T.
        x(t+1) = Ax(t)
    --------------------------------
    Parameters:
        A: State change matrix 
        x0: initialization state, array like.
        T: float, time for simulation
    '''
    n = len(x0)
    states = np.zeros((T, n))
    states[0] = x0 
    for i in range(1, T):
        states[i] = A @ states[i-1].T
    return states

def simulation_quadratic_control(A, B, x0, u, T=120):
    '''
    Simulation for a linear dynamical system with quadratic control, 
    with input u in time series t = 1, ..., T 
        x(t+1) = Ax(t) + Bu(t)
    --------------------------------
    Parameters:
        A: State change matrix 
        B: Input matrix
        u: matrix, 
        x0: initialization state, array like.
        T: float, time for simulation
    '''
    n = len(x0)
    states = zeros((T, n))
    states[0] = x0 
    t, m = u.shape
    if T > t:
        u = np.vstack([ u, zeros( (T-t-1, m) ) ]) # Expand input while T is greater than 
    for i in range(1, T):
        states[i] = A @ states[i-1] + B @ u[i-1]
    return states

def create_diag_matrix(A, T, yshift=None):
    '''
    Create a diagnal matrix with A, as 
        A ... ...
        ... A ...
        ... ... A
    ------------------------
    A: matrix 
    T: repeat time, int
    '''
    n, m = A.shape
    if not yshift:
        yshift = m

    ans = zeros( (n*T, yshift*(T-1)+m) )
    for i in range(T):
        ans[i*n:i*n+n, i*yshift:i*yshift+m] = A 
    return ans

def quadratic_control(A, B, C, x0, rho=100, T=100):
    n, m, q = A.shape[0], B.shape[1], C.shape[0]
    big_A = np.block([
        [create_diag_matrix(C, T), zeros((T*q, (T-1)*m))],
        [zeros(((T-1)*m, T*n)), sqrt(rho) * np.eye((T-1)*m)]
    ])

    big_C = np.block([
        [ create_diag_matrix(np.hstack([A, -np.eye(n)]), T-1, n), create_diag_matrix(B, T-1) ],
        [ np.hstack( [np.eye(n), zeros((n, n*(T-1)))] ), np.zeros((n,m*(T-1))) ],
        [ np.hstack( [zeros((n, n*(T-1))), np.eye(n)] ), np.zeros((n,m*(T-1))) ]
    ])

    d = np.hstack([ zeros(n*(T-1)), x0, zeros(n)])

    X, z = least_square_constrained(big_A, zeros(len(big_A)), big_C, d)
    x, u = X[:n*T].reshape(-1, n), X[n*T:].reshape(-1, m)
    return x, u

def estimate_K(A, B, C, rho=100, T=100):
    n, m = A.shape[0], B.shape[1]
    K = zeros( (n,m) )
    for (i, x0) in enumerate(np.eye(n)):
        _, u = quadratic_control(A, B, C, x0, rho=rho, T=T)
        K[i] = u[0]
    return K.T

def open_loop_plot(A, x0, T=120, save=False):
    # ---------- Open-loop Simulation --------------# 
    states = simulation_open_loop(A, x0, T)
    fig, ax = plt.subplots(2,2,figsize=(9,8), tight_layout=True)
    t = linspace(1, T+1, T)
    ax[0,0].plot(t, states[:,0], label='open-loop'); ax[0,0].set_ylabel("velocity along body axis $(x_t)_1$")
    ax[0,1].plot(t, states[:,1], label='open-loop'); ax[0,1].set_ylabel("velocity perpendicular to body axis $(x_t)_2$")
    ax[1,0].plot(t, states[:,2], label='open-loop'); ax[1,0].set_ylabel("angle of the body axis $(x_T)_3$")
    ax[1,1].plot(t, states[:,3], label='open-loop'); ax[1,1].set_ylabel("derivative of the angle of the body axis $(x_t)_4$")
    if save:
        fig.savefig('17-6-1.pdf')
    return fig, ax

def quadratic_plot(A, B, C, x0, T=120, save=False):
    fig, ax = open_loop_plot(A, x0, T)
    _, u = quadratic_control(A, B, C, x0, rho=rho, T=100)
    states = simulation_quadratic_control(A, B, x0, u)
    t = linspace(1, T+1, T)
    ax[0,0].plot(t, states[:,0], label='quadratic control'); 
    ax[0,1].plot(t, states[:,1], label='quadratic control'); 
    ax[1,0].plot(t, states[:,2], label='quadratic control'); 
    ax[1,1].plot(t, states[:,3], label='quadratic control'); 

    if save:
        for axx in ax.reshape(1,-1)[0]:
            axx.legend()
        fig.savefig('17-6-2.pdf')

    fig2, ax2 = plt.subplots(ncols=2, figsize=(9,4), tight_layout=True)
    if T > len(u) :
        u = np.vstack([u, zeros((T-len(u)-1, 2))])
    t = linspace(1, T, T-1)
    ax2[0].plot(t, u[:,0], label='optimal'); ax2[0].set_ylabel('Elevator angel $(u_t)_1$')
    ax2[1].plot(t, u[:,1], label='optimal'); ax2[1].set_ylabel('Engine thrust $(u_t)_2$')
    if save:
        for axx in ax2:
            axx.legend()
        fig.savefig('17-6-2(b).pdf')
    return fig, ax, fig2, ax2 

def simulation_feedback(A, B, K, x0, T=120):
    '''
    Simulation for a linear dynamical system with feedback gain matrix K 
    with input u in time series t = 1, ..., T 
        u(t) = Kx(t)
        x(t+1) = Ax(t) + Bu(t)
    --------------------------------
    Parameters:
        A: State change matrix 
        B: control matrix
        K: Feedback gain matrix
        x0: initialization state, array like.
        T: float, time for simulation
    '''
    n = len(x0)
    m, _ = K.shape
    states = np.zeros((T, n))
    states[0] = x0
    inputs = np.zeros((T-1, m))
    for i in range(1, T):
        inputs[i-1] = K @ states[i-1]
        states[i] = A @ states[i-1] + B @ inputs[i-1]
    return states, inputs

def feedback_plot(A, B, K, x0, T=120, rho=100, save=False):
    fig, ax, fig2, ax2 = quadratic_plot(A, B, C, x0)
    states, inputs = simulation_feedback(A, B, K, x0)
    t = linspace(1, T+1, T)
    ax[0,0].plot(t, states[:,0], label='feedback control'); 
    ax[0,1].plot(t, states[:,1], label='feedback control'); 
    ax[1,0].plot(t, states[:,2], label='feedback control'); 
    ax[1,1].plot(t, states[:,3], label='feedback control'); 

    if save:
        for axx in ax.reshape(1,-1)[0]:
            axx.legend()
        fig.savefig('17-6-3.pdf')
    
    t = linspace(1, T, T-1)
    ax2[0].plot(t, inputs[:,0], label='state feedback'); 
    ax2[1].plot(t, inputs[:,1], label='state feedback'); 
    if save:
        for axx in ax2:
            axx.legend()
        fig2.savefig('17-6-3(b).pdf')
    return fig, ax, fig2, ax2 
    
A = array([
    [0.99, 0.03, -0.02, -0.32],
    [0.01, 0.47, 4.70, 0.00],
    [0.02, -0.06, 0.40, 0.00],
    [0.01, -0.04, 0.72, 0.99]
])
B = array([
    [0.01, 0.99],
    [-3.44, 1.66],
    [-0.83, 0.44],
    [-0.47, 0.25]
])
C = np.eye(4)
rho = 100
x0 = array([0,0,0,1])
T=120

K = estimate_K(A, B, C, rho=rho, T=20)
print(K)
feedback_plot(A, B, K, x0, save=True)

#states, u = simulation_feedback(A, B, K, x0, T)
#plt.plot(states[:,0])

#states, u = quadratic_control(A, B, C, x0, rho, T)
#plt.plot(states[:,0])
plt.show()

def test():
    A=array([
        [0.855, 1.161, 0.667],
        [0.015, 1.073, 0.053],
        [-0.084, 0.059, 1.022]
    ])
    B = array([[-0.076], [-0.139], [0.342]])
    C = array([[0.218, -3.597, -1.683]])
    rho = 1
    x0 = array([0.496, -0.745, 1.394])
    T = 100

    x, u = quadratic_control(A, B, C, x0, rho)
    print(x.shape, u.shape)

    K = estimate_K(A, B, C, rho=rho, T=150)
    states, u = simulation_feedback(A, B, K, x0, T=150)
    y = [C @ state for state in states]
    plt.plot(y) 

    x, u = quadratic_control(A, B, C, x0, rho=1, T=100)
    states = simulation_quadratic_control(A, B, x0, u, T=150)
    y = [C @ state for state in states]
    #plt.plot(u)
    plt.plot(y)
    #open_loop_plot(A, x0, T=T)
    #quadratic_plot(A, B, C, x0, T=T)
    #K = estimate_K(A, B, C, rho=rho, T=150)
    print(K)
    #feedback_plot(A, B, K, x0, T, rho)
    plt.show()


    #print(K)
    #feedback_plot(A, B, K, x0, save=False)