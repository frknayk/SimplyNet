import numpy as np

from Functionals.activation import Activations

class HiddenLayer:
    def __init__(self,layer_dict,learning_rate):
        self.num_of_neurons = layer_dict['hidden_size']
        self.layer_type = layer_dict['activation_fnc']
        self.learning_rate = learning_rate
        self.activation_fnc = Activations(self.layer_type)
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
    
    def init_layer(self,n_x,init_coeff=0.01):
        self.W = np.random.randn(self.num_of_neurons,n_x)*init_coeff
        self.b = np.zeros( (self.num_of_neurons,1) )

    def forward(self,A):
        Z = np.dot(self.W,A) + self.b
        A = self.activation_fnc.f(Z)
        return A
    
    def backward(self,dA,cache):
        activation_cache,linear_cache = cache
        dZ = self._backward_activation(dA,activation_cache)
        dA_prev = self._backward_linear(dZ,linear_cache)
        return dA_prev

    def _backward_activation(self,dA, activation_cache):
        dZ = self.activation_fnc.f_dot(dA,activation_cache)
        return dZ

    def _backward_linear(self,dZ,linear_cache):
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        
        self.dW = (1/m)*np.dot(dZ,A_prev.T)
        self.db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        self.dA_prev = np.dot(np.transpose(W),dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return self.dA_prev

    def _update(self):
        """ Gradient descent """
        self.W -= self.learning_rate*self.dW
        self.b -= self.learning_rate*self.db
