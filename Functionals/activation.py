import numpy as np

class Activations:
    def __init__(self,fnc_name):
        self.fnc_type = fnc_name
        self.f = None
        self.f_dot = None
        self.__select_fnc()

    def __select_fnc(self):
        if self.fnc_type == 'sigmoid' or self.fnc_type == 'logistic':
            self.f = self.__sigmoid
            self.f_dot = self.__sigmoid_der

        elif self.fnc_type == 'tanh' :
            self.f = self.__tanh
            self.f_dot = self.__tanh_der
        
        elif self.fnc_type == 'relu' or  self.fnc_type == "RELU" :
            self.f = self.__relu
            self.f_dot = self.__relu_der
            
        elif self.fnc_type == 'final':
            self.f = None
            self.f_dot = None

    def __sigmoid(self,Z):
        return 1 / (1 + np.exp(-Z))
    
    def __tanh(self,Z):
        return np.tanh(Z)

    def __relu(self,Z):
        return np.maximum(0,Z)

    def __sigmoid_der(self,dA,cache):
        """dZ[L] = da[L] * g'[l](Z[l]) where g'(x) = sigmoid(x)*(1-sigmoid(x)) """
        Z = cache
        g_dot = self.__sigmoid(Z)*( 1- self.__sigmoid(Z) ) 
        dZ = dA * g_dot
        return dZ

    def __tanh_der(self,dA,cache):
        """dZ[L] = da[L] * g'[l](Z[l]) where g'(x) = 1 - tanh(x)^2 """
        Z = cache
        g_dot = 1.0 - np.tanh(Z)**2
        dZ  =  dA * g_dot
        return dZ

    def __relu_der(self,dA,cache):
        """dZ[L] = da[L] * g'[l](Z[l]) where g'(x) = 0 if x<0  and 1 if x>0"""
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
