import numpy as np
import tinn.activations as activations

class Layer:

    def __init__(self,num_neurons,activation=None):
        """
            Creates a layer object
            Parameters:
                num_neurons (int)   : Number of neurons in this layer
                activation  (str)   : Activation function associated with this layer
        """
        self.num_neurons=num_neurons
        self.activation=activation
        self.bias=2*np.random.rand(self.num_neurons,1)-1
        self.weights=[]
        self.output=None

    def feed(self,inputs):
        """
            Feed the inputs throught this layer
            Parameters:
                inputs (array): List of inputs
            Returns:
                outputs(array): Output from the layer after feeding inputs
        """
        if len(self.weights)<=0:
            #If weights are not initialized , then give random weights
            self.weights=2*np.random.rand(self.num_neurons,len(inputs))-1
        #Do rest of feed forward here
        output=np.dot(self.weights,inputs)+self.bias
        output=activations.apply(output,self.activation)
        self.output=output
        return output




