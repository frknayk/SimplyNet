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

if __name__ == "__main__":
    ########################## READ DATA ############################
    from Data_readers.catvnoncat_reader import load_data,validate_load
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data('/home/furkan/Furkan/Codes/Coursera-DL/SimplyNet/Data/catvnoncat')

    # Explore your dataset 
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]


    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    ########################## Neural Network Intitation ##########################
    fcnn = FullyConnectedNetwork(input_size=12288,output_size=1,seed=1,arch_path="Configs/example_network.yaml")

    ########################## TRAINING ############################
    losses = []
    num_iterations = 2500
    for idx in range(0, num_iterations):
        # Neural network's prediction
        y_hat = fcnn.forward(X=train_x)

        loss = fcnn.cost(Y=train_y,y_hat=y_hat)
        losses.append(loss)
        
        fcnn.backward()

        fcnn.update()

        if idx % 100 == 0:
            print("training step : {0} -- loss : {1}".format(idx,loss))


    

