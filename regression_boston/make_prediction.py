"""
making prediction: bnn vs traditional
"""
import numpy as np
import matplotlib.pyplot as plt
import bnn_boston as bnn
import pickle
import pandas as pd
import layers
import corner
# loading data
data = pd.read_csv('../data/BostonHousing.csv')
train = data[:-10]
test = data[-10:]
Y = np.array(train['medv']).reshape(-1,1)
#X = rescale_me(np.array(train.drop('medv', axis=1)))
X = np.array(train.drop('medv', axis=1))
validate_Y = np.array(test['medv']).reshape(-1,1)
#validate_X = rescale_me(np.array(test.drop('medv', axis=1)))
validate_X = np.array(test.drop('medv', axis=1))

input_size = X.shape[1]
hidden_size = 10
output_size = Y.shape[1]


L1 = layers.InputLayer(X)
L2 = layers.FullyConnectedLayer(input_size,hidden_size)
L3 = layers.ReLuLayer()
L4 = layers.FullyConnectedLayer(hidden_size, output_size)
L5 = layers.LinearLayer()
L6 = layers.SquaredError()

layerz = [L1, L2, L3, L4, L5, L6]


with open("weights_bnn_boston.pkl", "rb") as fp:
    params = pickle.load(fp)


(w1, sw1), (b1, sb1), (w2, sw2), (b2, sb2) = params

layerz = bnn.update_layerz(layerz, w1, b1, w2, b2)

print(w1.mean(), sw1.mean(), b1.mean(), sb1.mean(), w2.mean(), sw2.mean(), b2.mean(), sb2.mean())



def nn_predict():
    "forward is same in nn and bnn"
    nn_prediction = bnn.forward(layerz, validate_X[0])
    print("predict =", nn_prediction, "truth=", validate_Y[0])


def bnn_predict():
    """
    Make predictions using a trained Bayesian Neural Network (BNN).
    """

    layerz = [L1, L2, L3, L4, L5, L6]
    # Initialize arrays to store predictions
    predictions = []

    w1_ = w1.mean(axis=1)
    sw1_ = sw1.mean(axis=1)
    b1_ = b1[0]
    sb1_ = sb1[0]
    w2_ = w2[:,0]
    sw2_ = sw2[:,0]
    b2_ = b2[0]
    sb2_ = sb2[0]
    breakpoint()

    # Perform predictions using multiple weight samples
    for _ in range(1000): # we want 10000 samples
        # Sample weights from the trained BNN model
        #sampled_w1 = np.array([np.random.normal(w1_ , sw1_) for _ in range(hidden_size)]).reshape(input_size,hidden_size)
        #sampled_b1 = np.array([np.random.normal(b1_ , sb1_ ) for _ in range(hidden_size)])
        #sampled_w2 = np.array([np.random.normal(w2_ , sw2_ ) for _ in range(output_size)]).reshape(hidden_size, output_size)
        #sampled_b2 = np.array([np.random.normal(b2_ , sb2_ ) for _ in range(output_size)])

        sampled_w1 = np.array(np.random.normal(w1, sw1))
        sampled_b1 = np.array(np.random.normal(b1, sb1 ))
        sampled_w2 = np.array(np.random.normal(w2, sw2 ))
        sampled_b2 = np.array(np.random.normal(b2, sb2 ))
        layerz = bnn.update_layerz(layerz, sampled_w1, sampled_b1, sampled_w2, sampled_b2)
        # Make predictions with the sampled weights
        y_pred = bnn.forward(layerz, validate_X[1])

        # Store the predictions
        predictions.append(y_pred)

    # Calculate the mean and standard deviation of the predictions
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)

    #ndim, nsamples = 2, 10000
    #samples = np.random.normal(mean_predictions, std_predictions, ndim * nsamples).reshape([nsamples, ndim])
    figure = plt.figure()
    x_ = np.array(predictions).flatten()
    corner.corner(np.array([x_, x_]).T, fig=figure, color='red', label="pred")
    corner.corner(np.array([validate_Y, validate_Y]).T, fig=figure, color='blue', label="truth")
    plt.savefig('bnn_prediction.png')

    return mean_predictions, std_predictions


if __name__=="__main__":
    nn_predict()
    bnn_predict()
