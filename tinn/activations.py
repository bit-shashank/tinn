"""
    Implementation of activation functions along with
    their derivatives
"""

import math
import numpy as np


def sigmoid(x, deri=False):
    """
        Sigmoid activation function:
        Parameters:
            x    (array)  : A numpy array
            deri (boolean): If set to True function calulates the derivative of sigmoid
        Returns:
            x    (array)  : Numpy array after applying the approprite function
    """
    if deri:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


def relu(x,deri=False):
    if deri:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:
        return np.maximum(0,x)

def softmax(X, deri=False):
    if deri:
        return np.multiply( X, 1 - X ) + sum(
            # handle the off-diagonal values
            - X * np.roll( X, i, axis = 1 )
            for i in range(1, X.shape[1] )
        )
    else:
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo/expo_sum



def apply(array,activation,deriv=False):
    if activation=='sigmoid':
        return sigmoid(array,deriv)
    
    if activation=='relu':
        return relu(array,deriv)
    
    if activation=='softmax':
        return softmax(array,deriv)
        
    else:
        print("Invalid activation using linear") 
        return array