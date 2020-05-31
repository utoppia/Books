#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
# Python 3.6+
# Code for Exerises 3.22
# Author: Utoppia
# Date  : 02 May 2020

import numpy as np 
from numpy import linalg as LA 

class Position:
    def __init__(self, name, latitude, longtitude):
        self.name = name
        self.latitude = latitude
        self.longtitude = longtitude

def creat_coodinate(R, latitude, longtitude):
    return np.array(
        [
            R * np.sin(longtitude * np.pi / 180.0) * np.cos(latitude * np.pi / 180.0),
            R * np.cos(longtitude * np.pi / 180.0) * np.cos(latitude * np.pi / 180.0),
            R * np.sin(latitude * np.pi / 180.0)
        ])

def main():
    R = 6367.5
    b = Position('Beijing',   39.914, 116.392)
    p = Position('Palo Alto', 37.429, -122.138)

    bv = creat_coodinate(R, b.latitude, b.longtitude)
    pv = creat_coodinate(R, p.latitude, p.longtitude)

    # Distance through the earth
    dis_1 = LA.norm(bv-pv)
    print('Distance through the earth between %s and %s is %.2f km' % (b.name, p.name, dis_1))

    # Angle between two points
    angle = np.arccos(np.dot(bv, pv) / LA.norm(bv) / LA.norm(pv))

    # Distance along the surface of the earth
    dis_2 = R * angle 
    print('Distance along the surface of the earth between %s and %s is %.2f km' % (b.name, p.name, dis_2))

main()