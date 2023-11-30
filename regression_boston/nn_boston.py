""" ANN """
import layers
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
def main():
    """
    Architecture:
        Input -> FC (100 outputs)
              -> ReLU (generally good and has range [0, inf) which is desirable for images)
              -> FC (10 output, since this is last layer)
              -> Softmax (used for multiclass classification)
              -> Cross Entropy objective (used for multiclass classification)
    """
    eta = 1e-2
    epsilon = 1e-10
    epochs = 100
    s = 0
    r = 0
    rho1 = 0.1
    rho2 = 0.999
    delta = 10**-8

    hidden_size = 10
    data = pd.read_csv('../data/BostonHousing.csv')
    train = data[:-10]
    test = data[-10:]
    Y = np.array(train['medv']).reshape(-1,1)
    X = np.array(train.drop('medv', axis=1))
    validate_Y = np.array(test['medv']).reshape(-1,1)
    validate_X = np.array(test.drop('medv', axis=1))
    print("Y shape:", Y.shape)
    print("X shape:", X.shape)
    print("validate Y shape:", validate_Y.shape)
    print("validate X shape:", validate_X.shape)


    # first FC -> ReLU
    sizeIn = X.shape[1]
    sizeOut = hidden_size #Y.shape[1]
    W = np.random.normal(0, 1., (sizeIn, sizeOut))
    L1 = layers.InputLayer(X)
    L2 = layers.FullyConnectedLayer(sizeIn,sizeOut)
    L2.setWeights(W)
    L2.setBiases(np.random.uniform(0,1,sizeOut))

    L3 = layers.ReLuLayer()

    # second FC -> Softmax
    sizeIn = sizeOut
    sizeOut = 1 #Y.shape[1] #Y.shape[1]
    W = np.random.normal(0, 1., (sizeIn, sizeOut))
    L4 = layers.FullyConnectedLayer(sizeIn,sizeOut)
    L4.setWeights(W)
    L4.setBiases(np.random.uniform(0,1, sizeOut))

    L5 = layers.LinearLayer()


    L6 = layers.SquaredError()

    layerz = [L1, L2, L3, L4, L5, L6]

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
                layerz[i].updateWeights(grad, eta)#, adam=True, t=t+1)
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
    plt.savefig("nn_boston.png")


    weights = [np.array(layerz[1].getWeights()), np.array(layerz[1].getBiases()),
               np.array(layerz[3].getWeights()), np.array(layerz[3].getBiases())]

    print("weights means, std")
    with open("weights_nn_boston.pkl", "wb") as fp:
        pickle.dump(weights, fp)

    return MSEs


if __name__=="__main__":
    main()
