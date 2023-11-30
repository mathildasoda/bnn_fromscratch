"""
Running traditional neural network on MNIST
"""
import layers
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
def main():
    """
    Architecture:
        Input -> FC
              -> ReLU
              -> FC
              -> Softmax
              -> Cross entropy
    """
    eta = 1e-2
    epsilon = 1e-10
    epochs = 100 #1000
    s = 0
    r = 0
    rho1 = 0.1
    rho2 = 0.999
    delta = 10**-8


    data = pd.read_csv('../data/mnist_train_100.csv')
    Y_notEncoded = np.array(data['0'])
    Y = np.eye(10)[Y_notEncoded] # one hot encoded Y
    #Y = Y.reshape(Y.shape[0],1)
    X = np.array(data[data.columns.to_list()[1:]])
    validate_set = pd.read_csv('../data/mnist_valid_10.csv')
    validate_Y_notEncoded = np.array(validate_set['0'])
    #validate_Y = validate_Y.reshape(validate_Y.shape[0],1)
    validate_Y = np.eye(10)[validate_Y_notEncoded] # one hot encoded Y
    validate_X = np.array(validate_set[validate_set.columns.to_list()[1:]])
    # Xavier's:
    print("Y shape:", Y.shape)
    print("X shape:", X.shape)
    print("validate Y shape:", validate_Y.shape)
    print("validate X shape:", validate_X.shape)

    # continue running for another 100 epochs with these pretrained weights
    with open("weights_nn_mnist.pkl", "rb") as fp:
        parameters = pickle.load(fp)

    w1, b1, w2, b2 = parameters


    # first FC -> ReLU
    sizeIn = X.shape[1]
    sizeOut = 100 #Y.shape[1]
    xavier = np.sqrt(6/(sizeIn+sizeOut))
    #print(6/(sizeIn+sizeOut))
    #W = np.random.uniform(-xavier, xavier, size=(sizeIn, sizeOut))
    #W = np.random.normal(0, np.sqrt(6/sizeIn), (sizeIn, sizeOut))
    #W = np.random.normal(0, 1., (sizeIn, sizeOut))
    L1 = layers.InputLayer(X)
    L2 = layers.FullyConnectedLayer(sizeIn,sizeOut)
    L2.set_adam_param(s, r, rho1, rho2, delta)
    #L2.setWeights(W)
    #L2.setBiases(np.random.sample(sizeOut))
    #L2.setBiases(np.random.uniform(0,1,sizeOut))
    L2.setWeights(w1)
    L2.setBiases(b1)

    L3 = layers.ReLuLayer()

    # second FC -> Softmax
    sizeIn = sizeOut
    sizeOut = Y.shape[1] #Y.shape[1]
    xavier = np.sqrt(6/(sizeIn+sizeOut))
    #W = np.random.uniform(-xavier, xavier, size=(sizeIn, sizeOut))
    #print(6/(sizeIn+sizeOut))
    #W = np.random.normal(0, np.sqrt(6/(sizeIn+sizeOut)), (sizeIn, sizeOut))
    #L4 = layers.InputLayer(X)
    W = np.random.normal(0, 1., (sizeIn, sizeOut))
    L4 = layers.FullyConnectedLayer(sizeIn,sizeOut)
    L4.set_adam_param(s, r, rho1, rho2, delta)
    #L4.setWeights(W)
    #L4.setBiases(np.random.sample(sizeOut))
    #L4.setBiases(np.random.uniform(0,1, sizeOut))
    L4.setWeights(w2)
    L4.setBiases(b2)

    L5 = layers.SoftmaxLayer()


    L6 = layers.CrossEntropy()
    #print("eval=", layer.eval(Y, Yhat))
    #print("gradient=", layer.gradient(Y, Yhat))

    layerz = [L1, L2, L3, L4, L5, L6]
    # forwards
    #Yhat = X @ W


    MSEs = []
    validation_accuracies = []
    train_accuracies = []

    for t in tqdm(range(epochs)):
        h = X

        #"forward"
        for j in range(len(layerz)-1):
            h = layerz[j].forward(h)

        # RMSE:
        Yhat = h
        MSEs.append(layerz[-1].eval(Y, Yhat))
        print("MSE:", MSEs[-1])
        if MSEs[-1]<epsilon:
            print("min MSE reached")
            break

        # backward
        grad = layerz[-1].gradient(Y, Yhat)

        for i in range(len(layerz)-2, 0, -1):
            newgrad = layerz[i].backward(grad)

            if (isinstance(layerz[i], layers.FullyConnectedLayer)):
                layerz[i].updateWeights(grad, eta, adam=True, t=t+1)
                #layerz[i].updateWeights(grad, eta)

            grad = newgrad

# Calculate training accuracy


        train = np.mean(np.argmax(Yhat, axis=1) == np.argmax(Y, axis=1))
        train_accuracies.append(train)

        # VALIDATIO
        h_validate = validate_X
        #"forward"
        for j in range(len(layerz)-1):
            h_validate = layerz[j].forward(h_validate)
        # RMSE:
        Yhat_validate = h_validate

        validation = np.mean(np.argmax(Yhat_validate, axis=1) == np.argmax(validate_Y, axis=1))
        validation_accuracies.append(validation)

        #validation_loss = layerz[-1].eval(validate_Y, Yhat_validate)
        #validation_losses.append(validation_loss)

    print("Final RMSE:", np.sqrt(MSEs[-1]))
    print("Final training accuracy:", train_accuracies[-1])
    print("Final validation accuracy:", validation_accuracies[-1])
    plt.scatter(np.arange(0, epochs, 1), train_accuracies, label="training vs epoch", marker='.')
    plt.scatter(np.arange(0, epochs, 1), validation_accuracies, label="validation vs epoch", marker='.')
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("nn_mnist.png")


    weights = [np.array(layerz[1].getWeights()), np.array(layerz[1].getBiases()),
               np.array(layerz[3].getWeights()), np.array(layerz[3].getBiases())]
    breakpoint()
    with open("weights_nn_mnist_100epochAfterPretrain.pkl", "wb") as fp:
        pickle.dump(weights, fp)

    return MSEs


if __name__=="__main__":
    main()
