import matplotlib.pyplot as plt
import numpy as np
from Networks.fcnn import FullyConnectedNetwork

# Initate neural network
fcnn = FullyConnectedNetwork(input_size=2,output_size=1,seed=1,arch_path="Examples/Weight_Loading_Example/tiny_network.yaml")

# TRAINING
train_x = np.random.rand(2,1000)
train_y = np.random.rand(1,1000)
for idx in range(0, 150):
    # Neural network's prediction
    y_hat = fcnn.forward(X=train_x)
    loss = fcnn.cost(Y=train_y,y_hat=y_hat)
    fcnn.backward()
    fcnn.update()

print("weights are saved")
fcnn.save_weights('Examples/random_weight_2_4_4_1')

