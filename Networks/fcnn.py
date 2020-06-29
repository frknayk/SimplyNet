import numpy as np

from Layer.layers import Layers

class FullyConnectedNetwork:
    def __init__(self,input_size,output_size,arch_path,seed=None):
        self.arch_path = arch_path
        self.layers = Layers(arch_path,input_size,output_size)
        
        if seed is not None:
            np.random.seed(seed)

    def forward(self,X):
        print("------------")
        self.layers.print_layer_shapes()
        out = X
        print(out.shape)
        for layer in self.layers.Layers:
            out = layer.forward(out)
            print(out.shape)
        print("------------")
        return out
        
    def backward(self,loss_func):
        pass

    def update(self):
        pass

    def zero_grad(self):
        pass
    
    def cost(self,y,y_hat):
        return 0

if __name__ == "__main__":
    from Data_readers.mnist_reader import Reader_MNIST

    mnist_data = Reader_MNIST("Data/mnist/")
    train_data = mnist_data.train_data # (60000, 784)
    labels_train = mnist_data.labels_train
    test_data = mnist_data.test_data # (10000, 784)
    labels_test = mnist_data.labels_test

    fcnn = FullyConnectedNetwork(input_size=28**2,output_size=10,arch_path="Configs/example_network.yaml")

    fcnn.zero_grad()

    losses = []

    for idx,img in enumerate(train_data):
        # Label of image
        y = labels_train[idx]

        # Neural network's prediction
        y_hat = fcnn.forward(img)

        # losses.append(fcnn.cost(y,y_hat))

        # fcnn.backward(loss_func=None)

        # fcnn.update()


    

