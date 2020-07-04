import numpy as np

from Functionals.activation import Activations

class HiddenLayer:
    def __init__(self,layer_dict,learning_rate,requires_grad=True,seed=0):
        self.num_of_neurons = layer_dict['hidden_size']
        self.layer_type = layer_dict['activation_fnc']
        self.learning_rate = learning_rate
        self.activation_fnc = Activations(self.layer_type)
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.linear_cache = None
        self.activation_cache = None
        self.requires_grad = requires_grad
        self.seed = seed
    
    def init_layer(self,n_x,init_coeff=0.01):
        np.random.seed(self.seed)
        self.W = np.random.randn(self.num_of_neurons,n_x)/np.sqrt(n_x)
        self.b = np.zeros( (self.num_of_neurons,1) )

    def forward(self,A):
        Z = self.W.dot(A) + self.b
        
        A_new = self.activation_fnc.f(Z)

        # Caching
        cache = None
        if self.requires_grad:
            self.linear_cache = (A,self.W,self.b)
            self.activation_cache = Z
            cache = (self.linear_cache,self.activation_cache)
        return A_new, cache
    
    def backward(self,dA,cache):
        linear_cache,activation_cache = cache
        dZ = self._backward_activation(dA,activation_cache)
        dA_prev = self._backward_linear(dZ,linear_cache)
        return dA_prev
    
    def update(self):
        self._update()

    def zero_grad(self):
        if self.dW is not None:
            self.dW = np.zeros( (self.dW.shape[0],self.dW.shape[1]) )
        if self.db is not None:
            self.db = np.zeros( (self.db.shape[0],self.db.shape[1]) )

    def _backward_activation(self,dA, activation_cache):
        dZ = self.activation_fnc.f_dot(dA,activation_cache)
        return dZ

    def _backward_linear(self,dZ,linear_cache):
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        
        self.dW = (1/m)*np.dot(dZ,A_prev.T)
        self.db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        self.dA_prev = np.dot(np.transpose(W),dZ)

        assert (self.dA_prev.shape == A_prev.shape)
        assert (self.dW.shape == W.shape)
        assert (self.db.shape == b.shape)

        return self.dA_prev

    def _update(self):
        """ Gradient descent """
        self.W = self.W - self.learning_rate*self.dW
        self.b = self.b - self.learning_rate*self.db
