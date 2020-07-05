import numpy as np

from Layer.layers import Layers

class FullyConnectedNetwork:
    def __init__(self,input_size,output_size,arch_path,seed=None,requires_grad=True):
        # Network architecture
        self.arch_path = arch_path
        
        # If true, store caches
        self.requires_grad = requires_grad
        
        # Initate layers
        self.layers = Layers(arch_path,input_size,output_size,self.requires_grad,seed=seed)
        
        # NN last layer derivative w.r.t loss function
        self.dA = None
        
    def forward(self,X):
        output = self.layers.forward(X)
        return output

    def backward(self):
        self.layers.backward(self.dA)

    def update(self):
        self.layers.update()

    def zero_grad(self):
        self.layers.zero_grad()
    
    def cost(self,Y,y_hat):
        cost = -np.sum( Y*np.log(y_hat) + (1-Y)*np.log(1-y_hat) ) * ( 1/Y.shape[1] )

        ### calc dA ###
        self.dA = - (np.divide(Y, y_hat) - np.divide(1 - Y, 1 - y_hat))

        return cost

    def save_weights(self,path):
        self.layers.save_weights(path)
    
    def load_weights(self,path):
        loaded_weights = self.layers.load_weights(path)
        return loaded_weights