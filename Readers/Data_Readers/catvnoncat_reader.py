import numpy as np
import h5py

def load_data(path):
    # train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    path_train = path + '/train_catvnoncat.h5'
    train_dataset = h5py.File(path_train, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    # test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    path_test = path + '/test_catvnoncat.h5'
    test_dataset = h5py.File(path_test, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def validate_load(train_x_orig,train_y,classes):
    import matplotlib.pyplot as plt
    # Example of a picture
    index = 10
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    plt.show()