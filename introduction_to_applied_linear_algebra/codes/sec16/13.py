#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 16.13
# Author: Utoppia
# Date  : 22 May 2020

import matplotlib.pyplot as plt 
import numpy as np 
from numpy.linalg import pinv
from PIL.Image import fromarray

class Operator:
    def __init__(self):
        pass
    def str2bytes(self, s, L = 16):
        # Extend string s to a L length string by padding white space in the tail of original string
        if len(s) > L:
            s = s[:L]
        else:
            s = s + (L-len(s))*' ' 
        byte = ''.join([format(x, '08b') for x in bytearray(s, 'utf-8')])
        ans = np.array([int(i) for i in byte], dtype=np.int8)
        ans = np.where(ans > 0, ans, -1) # change values of 0 to -1
        return ans
    def byte2str(self, byte, L=16):
        byte = np.where(byte > 0, byte, 0) # change values of -1 to 0
        ans = ''
        for i in range(L):
            val = chr(int(''.join([str(x) for x in byte[8*i:8*i+8]]), 2))
            ans += val
        ans = ans.strip()
        return ans
    def test(self):
        s = 'hello'
        mes = self.str2bytes(s)
        print(mes)
        s_1 = self.byte2str(mes)
        print(s_1)

class Image:
    def __init__(self, filename):
        self.img = plt.imread(filename)
        self.shape = self.img.shape 
        print(self.shape)
    def create_D(self, k):
        '''
        Parameters:
            k -> int: the length of the message
        '''
        with open('data-13.npy', 'rb') as f:
            self.D = np.load(f) 
            self.D_inv = np.load(f) 
        #self.D = np.random.choice([-1,0,1], (k*8, self.shape[0] * self.shape[1]))
        
        #self.D_inv = pinv(self.D)
        #with open('data-13.npy', 'wb') as f:
        #    np.save(f, self.D)
        #    np.save(f, self.D_inv)
        return self.D_inv, self.D

    def calc_z(self, alpha, s, D_inv, D):
        '''
        calc the estimte of modification 
        ---------------------
        Parameters:
            alpha -> float: model parameter, a positive value
            s -> array like: message vector 
            D -> array like: Decode matrix
            D_inv -> array like: pseudo-inverse of D
        '''
        z = D_inv.dot(alpha*(s.reshape(-1,1)) - D.dot(self.img.reshape(-1,1)))
        #print(np.linalg.norm(z))
        self.decode_img = self.img.reshape(-1,1) + z 
        self.decode_img = np.where(self.decode_img < 0, 0, self.decode_img) # change negative to be 0
        self.decode_img = np.where(self.decode_img > 1, 1, self.decode_img) # change greater than 1 to be 1
        
        self.decode_img = self.decode_img.reshape(self.shape)
        #plt.imsave('16-13-new(alpha={}).png'.format(alpha), self.decode_img, cmap='gray')
        #plt.imsave('16-13-dif(alpha={})f.png'.format(alpha), self.decode_img-self.img, cmap='gray')
        message = D.dot(self.decode_img.reshape(-1,1)) / alpha  
        message = np.sign(message)
        message = message.reshape(1,-1).astype('int')[0]
        return message, np.sqrt(np.linalg.norm(z))

def main():
    img = Image('16-13-original.png')
    operator = Operator()
    K = 16 # a message with length 16
    print("Randomly generate D and calc pseodu-inverse of D...")
    D_inv, D = img.create_D(K)
    print('Over.\n')
    zp = []
    alphas = np.linspace(10, 10000, 100)
    for alpha in alphas:
        message, z = img.calc_z(alpha, operator.str2bytes('I love you', K), D_inv, D)
        message = operator.byte2str(message, K)
        zp.append(z)
        print(alpha, message)

    # plot
    #print(zp)
    fig = plt.figure()
    plt.semilogx(alphas, zp)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$||z||^2$')
    plt.title(r'Norm of $z$ with changes of $\alpha$')
    plt.grid(True)
    fig.savefig('16-13.pdf')
    plt.show()

if __name__ == '__main__':
    main()

    