import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_diagnostics(bnn_samples, param_name):

    def autocorrelation(samples, lag):
        mean = np.mean(samples)
        n = len(samples)
        acf = np.correlate(samples - mean, samples - mean, mode='full') / (n * np.var(samples))
        return acf[n - 1 + lag]

# Assuming 'bnn_samples' is your multidimensional array of HMC samples
    max_lag = 100  # Adjust as needed

# Compute ACF for each parameter or dimension separately
    acf_values = []


    for dimension in range(bnn_samples.shape[1]):
        acf_values_dim = [autocorrelation(bnn_samples[:, dimension], lag) for lag in range(max_lag)]
        acf_values.append(acf_values_dim)

    plt.figure(figsize=(10, 5))
    for dimension, acf_dim in enumerate(acf_values):
        plt.plot(acf_dim, label=f'Dimension {dimension}')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title(f'Autocorrelation Function for all Dimension')
    plt.legend()
    plt.savefig(f'autocorr_{param_name}_bnn_mnist.png')
# Plot ACF for each parameter or dimension
#    plt.figure(figsize=(10, 5))
#    for dimension, acf_dim in enumerate(acf_values):
#        plt.plot(acf_dim)
#        plt.xlabel('Lag')
#        plt.ylabel('Autocorrelation')
#        plt.title(f'Autocorrelation Function for Dimension {dimension}')
#    plt.legend()
#    plt.savefig(f'autocorr_{param_name}_bnn_boston.png')
#
# 'samples' is your list or array of MCMC samples
    samples = bnn_samples
#    plt.figure(figsize=(10, 5))
#    for i in range(samples.shape[1]):  # Plot each parameter separately
        #plt.subplot(samples.shape[1], 1, i+1)
#        plt.plot(samples[:, i])
#        plt.xlabel('Iteration')
#        plt.ylabel(f'Parameter {i+1}')
#    plt.tight_layout()
#    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(samples.shape[1]):  # Plot each parameter separately
        plt.plot(samples[:, i], label=f"parameter {i+1}")
        #plt.plot(acf_dim, label=f'Dimension {dimension}')
        plt.xlabel('Iter')
        plt.ylabel(f'Parameter')
    plt.tight_layout()
    plt.savefig(f'individualSamples_{param_name}_bnn_mnist.png')

if __name__=="__main__":
    with open("weights_bnn_mnist.pkl", "rb") as fp:
        f = pickle.load(fp)

    # just plotting means of weights
    w1 = f[0][0]
    w2 = f[2][0]

    plot_diagnostics(w1, "w1")
    plot_diagnostics(w2, "w2")
