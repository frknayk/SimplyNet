import matplotlib.pyplot as plt
from Networks.fcnn import FullyConnectedNetwork

# Import dataset readers
from Readers.Data_Readers.mnist_reader import Reader_MNIST

mnist_reader = Reader_MNIST("/home/furkan/Furkan/Codes/Coursera-DL/SimplyNet/Data/mnist/")

# Initate neural network
fcnn = FullyConnectedNetwork(input_size=28**2,output_size=10,seed=1,arch_path="Examples/Mnist/example_network.yaml")


# TRAINING
losses = []
num_iterations = 2500
for idx in range(0, num_iterations):
    # Neural network's prediction
    y_hat = fcnn.forward(X=mnist_reader.train_data )
    loss = fcnn.cost(Y=mnist_reader.labels_train,y_hat=y_hat)
    losses.append(loss)
    
    fcnn.backward()
    fcnn.update()
    if idx % 100 == 0:
        print("training step : {0} -- loss : {1}".format(idx,loss))

# plot the cost
plt.plot(losses)
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(fcnn.layers.lr))
plt.show()


