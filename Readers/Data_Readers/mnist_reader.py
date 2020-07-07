import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Reader_MNIST:
    def __init__(self,path):
        self.img_size = 28
        self.num_of_labels = 10

        # Relative path of mnist dataset
        self.path = path

        self.train_data = None
        self.train_data_all = None
        self.test_data = None
        self.test_data = None
        self.labels_train = None
        self.labels_test = None

        print("Loading mnist dataset, it may take a few minutes ...")
        self.read()
        self.parse()
        self.label_one_hot_encode()
        print("Mnist dataset is loaded succesfully !")

    def read(self):
        """Read and parse mnist dataset as train/test 
        (dataset downloaded from : https://www.python-course.eu/neural_network_mnist.php)
        """
        train_path = self.path + "mnist_train.csv"
        test_path = self.path + "mnist_test.csv"
        # Read training set        
        self.train_data_all = pd.read_csv(train_path) 
        self.train_data_all = pd.DataFrame(self.train_data_all).to_numpy()
        
        # Read test set
        self.test_data_all = pd.read_csv(test_path) 
        self.test_data_all = pd.DataFrame(self.test_data_all).to_numpy()
        # self.train_data_all = np.loadtxt(self.path + "mnist_train.csv", 
        #                     delimiter=",")

        # self.test_data_all = np.loadtxt(self.path + "mnist_test.csv", 
        #                        delimiter=",") 
        
    def parse(self):
        self.train_data = self.train_data_all[:,1:]
        self.labels_train = self.train_data_all[:,0]
        del self.train_data_all

        self.test_data = self.test_data_all[:,1:]
        self.labels_test = self.test_data_all[:,0]
        del self.test_data_all

        # According to the convention of SimplyNet
        # first dim must be shape of input
        # second dim must be number of samples (or batch)
        self.train_data = self.train_data.T
        self.test_data = self.test_data.T

    def label_one_hot_encode(self):
        self.label_one_hot_encode_train()
        self.label_one_hot_encode_test()

    def label_one_hot_encode_train(self):
        labels_train_new = np.zeros((self.num_of_labels,self.labels_train.shape[0]))
        for x in range( self.labels_train.shape[0] ) :
            labels_train_new[:,x] = one_hot( int(self.labels_train[x]) ,10)
        self.labels_train = labels_train_new

    def label_one_hot_encode_test(self):
        labels_test_new = np.zeros((self.num_of_labels,self.labels_test.shape[0]))
        for x in range( self.labels_test.shape[0] ) :
            labels_test_new[:,x] = one_hot( int(self.labels_test[x]) ,10)
        self.labels_test = labels_test_new

    def show_random(self):
        idx = np.random.randint(0,self.test_data.shape[0])
        label_encoded = np.where(self.labels_test[:,idx] == 1)
        label = int(label_encoded[0])
        img = self.test_data[:,idx].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.title(label="label : {0}".format(str(int(label))) )
        plt.show()
    
def one_hot(number, num_classes):
    encoded = np.zeros((1,num_classes))
    encoded[0][number] = 1
    return encoded

if __name__ == "__main__":
    # "Example Usage of the reader"
    mnist_reader = Reader_MNIST("Data/mnist/")
    
    mnist_reader.show_random()
    # mnist_reader.show_random()
    # mnist_reader.show_random()
    # mnist_reader.show_random()
    # mnist_reader.show_random()
    
