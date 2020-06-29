import numpy as np

from Data_readers.networker_reader import Reader
from Layer.layer import HiddenLayer

class Layers:
    def __init__(self,path,input_size,output_size):
        self.Layers = []
        self.reader = Reader(path)
        self.lr = self.reader.learning_rate
        self.create_layers(n_x_initial = input_size,n_y=output_size)

    def create_layers(self,n_x_initial,n_y):
        network_dict = self.reader.layers_dict
        
        # Hidden layers
        n_x = n_x_initial

        for idx,layer_number in enumerate(network_dict):
            
            layer_dict = network_dict[layer_number]
            
            # Hidden layers
            if  idx != len(network_dict)-1:
                layer = HiddenLayer(layer_dict,self.lr)
                layer.init_layer(n_x)
                self.Layers.append(layer)
                n_x = layer.num_of_neurons
            
            # Last hidden layer is initated seperately
            # due to output size is given from outside
            else :
                layer_dict['hidden_size'] = n_y
                layer = HiddenLayer(layer_dict,self.lr)
                layer.init_layer(n_x)
                self.Layers.append(layer)



    def print_layer_shapes(self):
        print("Layer Hidden Weights Dimensions")
        for layer in self.Layers:
            print(layer.W.shape)
        print("****")

if __name__ == "__main__":
    layers = Layers("Configs/example_network.yaml",28**2,10)
    layers.print_layer_shapes()
    

    X = np.random.randn(28**2,)
    layer_1 = layers.Layers[0]
    layer_2 = layers.Layers[1]

    out_1 = layer_1.forward(X)
    out_2 = layer_2.forward(out_1)

    print("A[0] shape : ",X.shape)
    print("A[1] shape : ",out_1.shape)
    print("A[2] shape : ",out_2.shape)
