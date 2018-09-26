# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:27:52 2018
Pass in two known data points plus one intensive property, P , and then calculate the other intensive property using interpolation
@author: Lonk
"""
import numpy

def interp(x1,x2,y1,y2,P):
    """
    Pass in two known data points plus one intensive property, P , and then calculate the other intensive property using interpolation
    """
    y = ((y2-y1)/(x2-x1)) * (P-x1) + y1
    return y 