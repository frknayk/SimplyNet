import numpy as np
import matplotlib.pyplot as plt

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
        print("Mnist dataset is loaded succesfully !")

    def read(self):
        """Read and parse mnist dataset as train/test 
        (dataset downloaded from : https://www.python-course.eu/neural_network_mnist.php)
        """
        # Read training set        
        self.train_data_all = np.loadtxt(self.path + "mnist_train.csv", 
                            delimiter=",")
        # Read test set
        self.test_data_all = np.loadtxt(self.path + "mnist_test.csv", 
                               delimiter=",") 
        
    def parse(self):
        self.train_data = self.train_data_all[:,1:]
        self.labels_train = self.train_data_all[:,0]
        del self.train_data_all

        self.test_data = self.test_data_all[:,1:]
        self.labels_test = self.test_data_all[:,0]
        del self.test_data_all

    def show_random(self):
        idx = np.random.randint(0,self.test_data.shape[0])
        label = self.labels_test[idx]
        img = self.test_data[idx].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.title(label="label : {0}".format(str(int(label))) )
        plt.show()

if __name__ == "__main__":
    "Example Usage of the reader"
    mnist_reader = Reader_MNIST("data/mnist/")
    
    mnist_reader.show_random()
    mnist_reader.show_random()
    mnist_reader.show_random()
    mnist_reader.show_random()
    mnist_reader.show_random()
    
