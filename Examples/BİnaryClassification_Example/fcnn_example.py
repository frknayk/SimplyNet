import matplotlib.pyplot as plt
from Networks.fcnn import FullyConnectedNetwork

# Import dataset readers
from Readers.catvnoncat_reader import load_data,validate_load
train_x_orig, train_y, test_x_orig, test_y, classes = load_data('/home/furkan/Furkan/Codes/Coursera-DL/SimplyNet/Data/catvnoncat')


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

 # Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


# Initate neural network
fcnn = FullyConnectedNetwork(input_size=12288,output_size=1,seed=1,arch_path="Examples/BİnaryClassification_Example/example_network.yaml")

# TRAINING
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

# plot the cost
plt.plot(losses)
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(fcnn.layers.lr))
plt.show()


