"""
hw2 - mathilda

NOTE:
- all dataIn are casted as np.array()
"""

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    @abstractmethod
    def forward(self, dataIn):
        """
        dataInt: matrix
        ----------
        return: zscored version of dataInt
        """
        #zscore is computed using mean and std attr of data
        # stores the input and output in attres
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward(self, gradIn):
        """
        takes backcoming gradient
        returns: updated gradient to be backpropped
        """
        pass



class InputLayer(Layer):
    def __init__(self, dataIn):
        """
        input: dataIn, an NxD matrix
            Take in entire training set
            Initialize mean and std of features
            Stds that are zeros will be set to 1
        output: None
        """
        Layer.__init__(self)
        self.meanX = np.array(dataIn).mean(axis=0) #mean
        stdX = np.array(dataIn).std(axis=0, ddof=1)
        stdX[stdX==0]=1
        self.stdX = stdX #std


    def forward(self, dataIn):
        """
        input: dataIn, an NxD matrix
        output: an NxD matrix, the zscored data
        """
        self.setPrevIn(np.array(dataIn))
        self.setPrevOut((np.array(dataIn)-self.meanX)/self.stdX)
        return self.getPrevOut()


    def gradient(self):
        # do nothing. Returning zscored data
        return np.array(self.getPrevOut())

    def backward(self, gradIn):
        return np.array(self.getPrevOut()*gradIn)



#----------------------------------------
# ACTIVATION LAYERS
# (1) init it's constructor within it's own constructor
# (2) implement forward method that taike in DATA as param
# (3) set parnet class' previous input to DATA
# (4) compute output value. Set parents' class prev output to that
# (5) return 4
#----------------------------------------

class LinearLayer(Layer):
    # g(z) = z
    def __init__(self):
        Layer.__init__(self)

    def forward(self, dataIn):
        self.setPrevIn(np.array(dataIn))
        self.setPrevOut(dataIn)
        return self.getPrevOut()

    def gradient(self):
        outer_dimension = int(self.getPrevIn().shape[0])
        inner_dimension = int(self.getPrevIn().shape[1])
        return np.array([np.eye(inner_dimension) for _ in range(outer_dimension)])

    def backward(self, gradIn):
        return np.array(self.gradient()*gradIn)


class ReLuLayer(Layer):
    # g(x) = max(0,z)
    def __init__(self):
        Layer.__init__(self)

    def forward(self, dataIn):
        self.setPrevIn(np.array(dataIn))
        self.setPrevOut(np.maximum(0,dataIn))
        return self.getPrevOut()


    def gradient(self):
        return np.where(self.getPrevIn()>0, 1,0)

    def backward(self, gradIn):
        return np.array(self.gradient()*gradIn)


class LogisticSigmoidLayer(Layer):
    # g(z) = 1/(1+e^(-z))
    def __init__(self):
        Layer.__init__(self)

    def forward(self, dataIn):
        self.setPrevIn(np.array(dataIn))
        self.setPrevOut(1/(1+np.exp(-dataIn)))
        return self.getPrevOut()

    def gradient(self):
        return self.getPrevOut()*(1-self.getPrevOut())

    def backward(self, gradIn):
        return np.array(self.gradient()*gradIn)


class SoftmaxLayer(Layer):
    # g(z) = (e^z-max(z))/(SUM e^z_i-max(z))
    def __init__(self):
        Layer.__init__(self)

    def forward(self, dataIn):
        # numerically stable
        # kinda sus with the datain in denom
        self.setPrevIn(np.array(dataIn))
        y = np.exp(dataIn - np.max(dataIn))
        self.setPrevOut(y/np.sum(y, axis=0))
        return self.getPrevOut()

    def gradient(self):
        X = self.getPrevOut()
        g = lambda z : np.exp(z)/np.sum(np.exp(z))
        #grad = np.array([np.eye(X.shape[1], dtype=float) for _ in range(X.shape[0])])
        #print("grad is of shape:", grad.shape)

        #print("curr grad=\n", grad)
        #for outer in range(X.shape[0]):
        #    for i in range(X.shape[1]):
        #        for j in range(X.shape[1]):
        #            if i==j:
        #                g_z =g(X[outer][inner])
        #                grad[outer][i][j] = 1-g_z**2
        #print("returning grad=\n", grad)
        grad = np.array([np.diag(g(z))-g(z).T@g(z) for z in X])
        return np.array(grad)

    def backward(self, gradIn):
        return np.array(np.einsum('...i,...ij', gradIn, self.gradient()))




class TanhLayer(Layer):
    # g(z) = (e^z - e^(-z))/(e^z+e^-z)
    def __init__(self):
        Layer.__init__(self)

    def forward(self, dataIn):
        self.setPrevIn(np.array(dataIn))
        z = dataIn
        self.setPrevOut((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
        return self.getPrevOut()

    def gradient(self):
        return 1-self.getPrevOut()**2

    def backward(self, gradIn):
        return np.array(self.gradient()*gradIn)



#----------------------------------------
# FULLY CONNECTED LAYER
#----------------------------------------
class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        """
        sizeIn: num of features of data coming in
        sizeOut: num of features of data coming out
        W : sizeIn x sizeOut zero matrix
        bias: 1xsizeOut zero matrix
        """
        Layer.__init__(self)
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        #weights +biases are random values in the range of +-10^-4
        W = np.array([np.random.uniform(-10**-4, 10**-4) for i in range(sizeIn*sizeOut)])
        self.W = W.reshape(sizeIn, sizeOut)
        bias = np.array([np.random.uniform(-10**-4, 10**-4) for i in range(sizeOut)])
        self.bias = bias.reshape(1, sizeOut)

    def getWeights(self):
        """
        ----------
        return: sizeIn x sizeOut weight matrix
        """
        return self.W


    def setWeights(self, weights):
        """
        weights: sizeIn x sizeOut weight matrix
        """
        self.W = np.array(weights)


    def updateWeights(self, gradIn, eta, adam=False, t=0):
        #dJdb = np.sum(gradIn, axis=0)
        #dJdW = np.array((self.getPrevIn().T@gradIn))
        if not adam:
            dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
            dJdW = np.array((self.getPrevIn().T@gradIn))/gradIn.shape[0]
            if self.W.shape == dJdW.shape:
                self.W -= eta*np.array(dJdW)
            else:
                self.W -= eta*np.array(np.sum(dJdW, axis=0))
            self.bias -= eta*np.sum(dJdb)
        else:
                s, r, rho1, rho2, delta = self.get_adam_param()
                gradw = np.array((self.getPrevIn().T@gradIn))/gradIn.shape[0]
                s = rho1*s + (1-rho1)*gradw
                r = rho2*r+(1-rho2)*(gradw*gradw)
                curr_w = self.getWeights()
                curr_w -= eta*(s/(1-rho1**t))/(np.sqrt(r/(1-rho2**t))+delta)
                self.setWeights(curr_w)
    #def updateWeights(self, gradIn, eta):
    #    dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
    #    dJdW = np.array((self.getPrevIn().T@gradIn))/gradIn.shape[0]

        #if self.W.shape == dJdW.shape:
    #    self.W -= eta*np.array(dJdW)
        #else:
        #    self.W -= eta*np.array(np.sum(dJdW, axis=0))
    #    self.bias -= eta*np.sum(dJdb)


    def getBiases(self):
        """
        ----------
        return: 1 x sizeOut bias vector
        """
        return self.bias


    def setBiases(self, biases):
        """
        biases: 1 x sizeOut bias vector
        """
        self.bias = np.array(biases)

    def forward(self, dataIn):
        # XW +b
        self.setPrevIn(np.array(dataIn))
        self.setPrevOut(np.array(dataIn @ self.W + self.bias))
        return self.getPrevOut()

    def gradient(self):
        return self.W.T

    def backward(self, gradIn):
        grad = np.array(gradIn @ self.gradient())
        return grad

#----------------------------------------
# OBJECTIVE LAYER
#----------------------------------------
class SquaredError():
    def eval(self, Y, Yhat):
        return np.mean((Y-Yhat)*(Y-Yhat))

    def gradient(self, Y, Yhat):
        return -2*(Y-Yhat)


class LogLoss():
    def eval(self, Y, Yhat):
        epsilon = 1e-7
        return np.mean(-(Y*np.log(Yhat+epsilon)+(1-Y)*np.log(1-Yhat+epsilon)))

    def gradient(self, Y, Yhat):
        epsilon = 1e-7
        return -(Y-Yhat)/(Yhat*(1-Yhat)+epsilon)


class CrossEntropy():
    # numerically stable
    def eval(self, Y, Yhat):
        sumMe = []
        for i in range(Y.shape[1]):
            sumMe.append(Y.T[i]*np.log(Yhat.T[i]+1e-7))
        return -np.mean(np.sum(sumMe))

    def gradient(self, Y, Yhat):
        return -Y/Yhat
