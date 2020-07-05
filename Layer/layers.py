import numpy as np

from Readers.networker_reader import Reader
from Layer.layer import HiddenLayer
import pickle

class Layers:
    def __init__(self,path,input_size,output_size,requires_grad,seed):
        self.Layers = []
        self.reader = Reader(path)
        self.lr = self.reader.learning_rate
        self.seed = seed
        self.requires_grad = requires_grad
        self.create_layers(n_x_initial = input_size,n_y=output_size)
        self.init_layers(n_x_initial = input_size,n_y=output_size)
        self.caches = []

    def create_layers(self,n_x_initial,n_y):
        network_dict = self.reader.layers_dict
        
        # Hidden layers
        n_x = n_x_initial

        for idx,layer_number in enumerate(network_dict):
            
            layer_dict = network_dict[layer_number]
            
            # Hidden layers
            if  idx != len(network_dict)-1:
                layer = HiddenLayer(layer_dict,self.lr,self.requires_grad,seed=self.seed)
                self.Layers.append(layer)
                n_x = layer.num_of_neurons
            
            # Last hidden layer is initated seperately
            # due to output size is given from outside
            else :
                layer_dict['hidden_size'] = n_y
                layer = HiddenLayer(layer_dict,self.lr,seed=self.seed)
                layer.init_layer(n_x)
                self.Layers.append(layer)
    
    def init_layers(self,n_x_initial,n_y):
        """Initate all weights and biases with same bias !

        Args:
            n_x_initial (integer): input size of neural network
            n_y (integer): output size of neural network
        """
        np.random.seed(self.seed)
        
        n_x = n_x_initial

        for layer in self.Layers:
            layer.W = np.random.randn(layer.num_of_neurons,n_x)/np.sqrt(n_x)
            layer.b = np.zeros( (layer.num_of_neurons,1) )
            n_x = layer.num_of_neurons
            
    def forward(self,X):
        out = X
        for layer in self.Layers:
            out,cache = layer.forward(out)
            if self.requires_grad:
                self.caches.append(cache)
        return out

    def backward(self,dAL):
        dA = dAL

        for idx in reversed(range(len(self.Layers))):
            dA = self.Layers[idx].backward(dA,self.caches[idx])
            
    def update(self):
        for layer in self.Layers:
            layer.update()

        # At the end of every episode clear caches
        self.caches = []

    def zero_grad(self):
        for layer in self.Layers:
            layer.zero_grad()

    def save_weights(self,path):
        """Save layer weights with pickle as path+'.pickle'

        Args:
            path (string): full path to saved weights
        """
        file_name = path + ".pickle"
        layer_weights = []
        with open(file_name, 'wb') as handle:
            for idx,layer in enumerate( self.Layers ):
                layer_dict = {}
                layer_name = layer.layer_type + '_' + str(idx) 
                layer_dict[layer_name] = { 'W' : layer.W , 'b' : layer.b }
                layer_weights.append(layer_dict)
            
            pickle.dump(layer_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_weights(self,path):
        file_name = path + ".pickle"
        with open(file_name, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def load_weights(self,path):
        weights_list = self.get_weights(path)
        for idx,layer in enumerate( self.Layers ):
            layer_name = layer.layer_type + '_' + str(idx)
            layer_dict = weights_list[idx]
            weigts_dict = layer_dict[layer_name]
            W = weigts_dict['W']
            b = weigts_dict['b']
            layer.W = W
            layer.b = b

    def print_layer_shapes(self):
        print("Layer Hidden Weights Dimensions")
        for layer in self.Layers:
            print(layer.W.shape)
        print("****")
