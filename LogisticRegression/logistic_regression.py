import numpy as np
import os
import sys
import matplotlib.pyplot as plt

############### Add agent library to the path ###############
main_path = os.path.abspath(__file__)
index = main_path.find("/1_Neural_Nets_And_Deep_Learning")
main_dirname = main_path[:index] + "/1_Neural_Nets_And_Deep_Learning"
sys.path.append(main_dirname)
############################################################

from commons.activations import sigmoid
from commons.data_parser import *

class LogisticRegressor:
    def __init__(self,data,learning_rate=0.01,num_of_iterations=2000,print_cost=True):
        """Logistic Regression with sigmoid as activation function

        Args:
            data (dictionary): Contains train and test data together 
            learning_rate (float) 
            num_of_iterations (int, optional):Defaults to 2000.
            print_cost (bool, optional): Defaults to True.
        """
        self.X_train = data['train_set_input']
        self.Y_train = data['train_set_labels']
        self.X_test = data['test_set_input']
        self.Y_test = data['test_set_labels']
        self.classes = data['classes']
        
        # Learning rate
        self.lr = learning_rate
        
        # Number of training steps
        self.num_iterations = num_of_iterations

        # Debug mode
        self.print_cost = print_cost

        # Number of samples in training set
        self.m = None

        # Weights and bias
        self.w = None
        self.b = None
        
        self.init_params()

        # Cost trajectory
        self.costs = []

    def init_params(self):
        self.preprocess_data()
        self.w = np.zeros((self.X_train.shape[0],1))
        self.b = 0
        self.m = self.X_train.shape[1]
    
    def flatten_img(self,img):
        img = img.reshape(img.shape[0],-1).T
        return img

    def preprocess_data(self):
        "Normalize and flatten images"
        # # Flattening
        self.X_train = self.flatten_img(self.X_train)
        self.X_test = self.flatten_img(self.X_test)

        # self.X_train = self.X_train.reshape(self.X_train.shape[0],-1).T
        # self.X_test = self.X_test.reshape(self.X_test.shape[0], -1).T

        # Normalization
        self.X_train = self.X_train/255.
        self.X_test = self.X_test/255.

    def forward_propagation(self,X):
        A = sigmoid(np.dot(self.w.T,X)+self.b) 
        return A
    
    def backpropagation(self,X,Y):
        A = self.forward_propagation(X)
        
        cost = -np.sum( Y*np.log(A) + (1-Y)*np.log(1-A) )/self.m
        cost = np.squeeze(cost)

        # Derivative of cost w.r.t weight
        dw = np.dot(X,A.T-Y.T)/self.m
        # Derivative of cost w.r.t bias term
        db = np.sum(A-Y)/self.m

        return cost,dw,db

    def update_weights(self,dw,db):
        self.w = self.w - self.lr*dw
        self.b = self.b - self.lr*db

    def predict(self,X):
        """ Test model with new parameters"""
        A = self.forward_propagation(X)
        Y_prediction =  np.zeros((1,X.shape[1]))
        for i in range(A.shape[1]):
            if A[0,i] > 0.5:
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0   
        return Y_prediction

    def train(self):
        for i in range(self.num_iterations):
        
            # Calculate values out of activation function
            A  = self.forward_propagation(self.X_train)
            
            # Calculate cost and gradients
            cost,dw,db = self.backpropagation(self.X_train,self.Y_train)

            # Update weights
            self.update_weights(dw,db)

            # Record the costs
            self.costs.append(cost)

            # Print the cost every 100 training iterations
            if self.print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        
        self.plot_learning()

    def test(self):                
        # Predict test/train set examples
        Y_prediction_test = self.predict(self.X_test)
        Y_prediction_train = self.predict(self.X_train)

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(self.Y_train - Y_prediction_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - self.Y_test)) * 100))

    def plot_learning(self):
        plt.plot(self.costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.lr))
        plt.show()


if __name__ == "__main__":
    data = load_dataset(train_path="week_2/data/train_catvnoncat.h5",test_path="week_2/data/test_catvnoncat.h5")
    model = LogisticRegressor(data,num_of_iterations=2000)
    model.train()
    model.test()