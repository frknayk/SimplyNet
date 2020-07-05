import numpy as np

from Data_readers.networker_reader import Reader
from Layer.layer import HiddenLayer

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

    def print_layer_shapes(self):
        print("Layer Hidden Weights Dimensions")
        for layer in self.Layers:
            print(layer.W.shape)
        print("****")

if __name__ == "__main__":
    layers = Layers("Configs/example_network.yaml",12288,1)
    layers.print_layer_shapes()
    

    X = np.random.randn(12288, 209)
    layer_1 = layers.Layers[0]
    layer_2 = layers.Layers[1]
    layer_3 = layers.Layers[2]
    layer_4 = layers.Layers[3]

    out_1 = layer_1.forward(X)
    out_2 = layer_2.forward(out_1)
    out_3 = layer_3.forward(out_2)
    out_4 = layer_4.forward(out_3)

    print("Number of layers : {0}".format(len(layers.Layers)))
    print("A[0] shape : ",X.shape)
    print("A[1] shape : ",out_1.shape)
    print("A[2] shape : ",out_2.shape)
    print("A[3] shape : ",out_3.shape)
    print("A[4] shape : ",out_4.shape)
