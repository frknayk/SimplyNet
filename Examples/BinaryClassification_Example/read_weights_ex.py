from Networks.fcnn import FullyConnectedNetwork

# Initate neural network
fcnn = FullyConnectedNetwork(input_size=12288,output_size=1,seed=1,arch_path="Examples/BinaryClassification_Example/example_network.yaml")

# Relative path to saved weights
brain = fcnn.load_weights('Examples/cat_vs_notCat')


# Import dataset readers
from Readers.Data_Readers.catvnoncat_reader import load_data,validate_load
train_x_orig, train_y, test_x_orig, test_y, classes = load_data('/home/furkan/Furkan/Codes/Coursera-DL/SimplyNet/Data/catvnoncat')


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

 # Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


# Test if loss is around 0.0952.. (if so everything works great) 
y_hat = fcnn.forward(X=train_x)
loss = fcnn.cost(Y=train_y,y_hat=y_hat)
print("Loss : {0}".format(loss))