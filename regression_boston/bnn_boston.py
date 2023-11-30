import layers
#import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import copy
import scipy.stats as stats
from autograd import jacobian
import pickle
import autograd.numpy as np
from autograd import jacobian


# --------------------------------------------------------------------
# SETUPS
# --------------------------------------------------------------------

def rescale_me(arr):
    """ rescaled to [0,1] for easier training"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    rescaled_arr = (arr - min_val) / (max_val - min_val)
    return rescaled_arr

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
# Xavier's:
print("Y shape:", Y.shape)
print("X shape:", X.shape)
print("validate Y shape:", validate_Y.shape)
print("validate X shape:", validate_X.shape)


# --------------------------------------------------------------------
# HELPER
# --------------------------------------------------------------------
def plot_accuracies(train_accuracies, validation_accuracies):
    print("Final training accuracy:", train_accuracies[-1])
    print("Final validation accuracy:", validation_accuracies[-1])
    epochs = len(train_accuracies)
    plt.scatter(np.arange(0, epochs, 1), train_accuracies, label="training vs epoch", marker='.', alpha=0.5)
    plt.scatter(np.arange(0, epochs, 1), validation_accuracies, label="validation vs epoch", marker='.', alpha=0.5)
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("bnn_boston.png")
    plt.close()


def squash_params(params):
    """
    squashing parameters for easy sampling
    """
    (w1_mean, w1_std), (b1_mean, b1_std), (w2_mean, w2_std), (b2_mean, b2_std) = params
    norm_distrted = np.concatenate([w1_mean.flatten(), b1_mean.flatten(), w2_mean.flatten(), b2_mean.flatten()])
    unif_distrted = np.concatenate([w1_std.flatten(), b1_std.flatten(), w2_std.flatten(), b2_std.flatten()])
    return np.concatenate([norm_distrted, unif_distrted])


def unsquash_params(params, input_size=X.shape[1], hidden_size=10, output_size=Y.shape[1]):
    norm_distrted = params[:int(len(params)/2)]
    w1_mean = norm_distrted[:input_size*hidden_size].reshape(input_size, hidden_size)
    b1_mean = norm_distrted[input_size*hidden_size:input_size*hidden_size+hidden_size].reshape(hidden_size, )
    w2_mean = norm_distrted[input_size*hidden_size+hidden_size: input_size*hidden_size+hidden_size+hidden_size*output_size].reshape(hidden_size, output_size)
    b2_mean = norm_distrted[input_size*hidden_size+hidden_size+hidden_size*output_size:].reshape(output_size,)


    unif_distrted = params[int(len(params)/2):]
    w1_std = unif_distrted[:input_size*hidden_size].reshape(input_size, hidden_size)
    b1_std = unif_distrted[input_size*hidden_size:input_size*hidden_size+hidden_size].reshape(hidden_size, )
    w2_std = unif_distrted[input_size*hidden_size+hidden_size: input_size*hidden_size+hidden_size+hidden_size*output_size].reshape(hidden_size, output_size)
    b2_std = unif_distrted[input_size*hidden_size+hidden_size+hidden_size*output_size:].reshape(output_size,)

    return (w1_mean, w1_std), (b1_mean, b1_std), (w2_mean, w2_std), (b2_mean, b2_std)


# --------------------------------------------------------------------
# PROBABILITY FUNCTIONS
# --------------------------------------------------------------------

def calc_prior_logpdf(params): #TODO
    half_param = int(len(params) / 2)
    means = params[:half_param]
    stds = params[half_param:]

    # Gaussian prior for means (log-pdf)
    #mean_prior_logpdf = np.sum(stats.norm(loc=0, scale=1).logpdf(means))

    # Uniform prior for standard deviations (log-pdf)
    std_prior_logpdf = np.sum(-np.log(stds))  # Assuming uniform [0, inf)

    # Sum the log-pdfs of both components
    #prior_logpdf = mean_prior_logpdf #+ std_prior_logpdf
    mean_prior_logpdf = np.sum(stats.norm(loc=0, scale=1).logpdf(means))
    return mean_prior_logpdf + std_prior_logpdf


def calc_likelihood_logpdf(y_pred, target):
    likelihood_logpdf = np.sum(stats.norm(y_pred, 1).logpdf(target))
    return likelihood_logpdf



def calc_posterior_logpdf(params, y_pred, target):
    try:
        params = params._value
    except:
        params = params
    prior_logpdf = calc_prior_logpdf(params)
    likelihood_logpdf = calc_likelihood_logpdf(y_pred, target)
    posterior_logpdf = prior_logpdf + likelihood_logpdf
    return posterior_logpdf


def calc_kinetic_energy(p):
    return np.sum(p**2)/2


def calc_potential_energy(params, y_pred, target):
    """
    negative log likelihood of q (BNN parameters)
    """
    return -calc_posterior_logpdf(params, y_pred, target)


from autograd import grad, jacobian

def calc_grad_potential_energy(params, y_pred, target):
    """
    del U/ del q
    """
    def potential_energy(params):
        return -calc_posterior_logpdf(params, y_pred, target)
    #breakpoint()
    # Compute the gradient using Autograd
    out = jacobian(potential_energy)(params)
    return out
    #gradient = jacobian(potential_energy)(params).reshape(496, 302)
    #out = np.array([jacobian(potential_energy)(qi) for qi in params])
    #gradient_func = grad(potential_energy)
    #try:
    #    gradient = gradient_func(params)
    #    return np.gradient(log_post)
    #except TypeError:
    #    print("gradient faulty")
    #    return np.inf
    #breakpoint()
    #return np.sum(jacobian(potential_energy)(params).reshape(496, 302), axis=0)


# --------------------------------------------------------------------
# HAMILTONIAN MONTE CARLO
# --------------------------------------------------------------------

def run_HMC_sampler(
        init_bnn_params, # weights
        y_pred,
        target,
        n_hmc_iters=50,
        n_leapfrog_steps=50,
        step_size=3e-5,
        random_seed=42,
        calc_potential_energy=None,
        calc_kinetic_energy=None,
        calc_grad_potential_energy=None,
        ):
    """ Run HMC sampler for many iterations (many proposals)

    Returns
    -------
    bnn_samples : list
        List of samples of NN parameters produced by HMC
        Can be viewed as 'approximate' posterior samples if chain runs to convergence.
    info : dict
        Tracks energy values at each iteration and other diagnostics.

    References
    ----------
    See Neal's pseudocode algorithm for a single HMC proposal + acceptance:
    https://arxiv.org/pdf/1206.1901.pdf#page=14

    This function repeats many HMC proposal steps.
    """
    prng = np.random.RandomState(int(random_seed))

    # Create random-number-generator with specific seed for reproducibility
    start_time_sec = time.time()
    # Set initial bnn params
    cur_bnn_params = init_bnn_params
    cur_potential_energy = cur_bnn_params
    #posterior_grad = None
    bnn_samples = []
    # make lists to track energies over iterations
    accept_rate = 0.0
    n_accept = 0
    for t in range(n_hmc_iters):
        # Draw momentum for CURRENT configuration
        cur_momentum_vec = np.random.normal(0, 1, cur_bnn_params.shape)

        # Create PROPOSED configuration
        prop_bnn_params, prop_momentum_vec = make_proposal_via_leapfrog_steps(
            y_pred, target,
            cur_bnn_params, cur_momentum_vec,
            n_leapfrog_steps=n_leapfrog_steps,
            step_size=step_size,
            calc_grad_potential_energy=calc_grad_potential_energy)

        #if np.isinf(prop_bnn_params) or np.isinf(prop_momentum_vec):
        #    bnn_samples.append(cur_bnn_params)
        #    continue
        #Compute probability of accept/reject for proposal

        proposed_U = calc_potential_energy(prop_bnn_params, y_pred, target)
        proposed_K = calc_kinetic_energy(prop_momentum_vec)
        current_U = calc_potential_energy(cur_potential_energy, y_pred, target)
        current_K = calc_kinetic_energy(cur_momentum_vec)
        #print(proposed_U, proposed_K, current_U, current_K)
        H = np.exp(current_U - proposed_U + current_K - proposed_K)
        #print(H)
        accept_proba = min(1.0,H) # Metropolis

        # Draw random value from (0,1) to determine if we accept or not
        if prng.rand() < accept_proba:
            # If here, we accepted the proposal
            n_accept += 1


            #if posterior_grad is None:
            #    posterior_grad = calc_grad_potential_energy(prop_bnn_params)
            #else:
            #    posterior_grad += calc_grad_potential_energy(prop_bnn_params)

            #current state updated
            cur_bnn_params = prop_bnn_params
            cur_potential_energy = cur_bnn_params

        # Update list of samples from "posterior"
        bnn_samples.append(cur_bnn_params)
        # update energy tracking lists
        # Print some diagnostics every 50 iters
        if t < 5 or ((t+1) % 50 == 0) or (t+1) == n_hmc_iters:
            accept_rate = float(n_accept) / float(t+1)
            print("iter %6d/%d after %7.1f sec | accept_rate %.3f" % (
                t+1, n_hmc_iters, time.time() - start_time_sec, accept_rate))
    print("acceptance_rate=", accept_rate)
    return bnn_samples, accept_rate


def make_proposal_via_leapfrog_steps(
        y_pred, target,
        cur_bnn_params, cur_momentum_vec,
        n_leapfrog_steps=100,
        step_size=1.0,
        calc_grad_potential_energy=None):
    """ Construct one HMC proposal via leapfrog integration

    Returns
    -------
    prop_bnn_params : same type/size as cur_bnn_params
    prop_momentum_vec : same type/size as cur_momentum_vec

    """
    # Initialize proposed variables as copies of current values
    q = copy.deepcopy(cur_bnn_params)
    p = copy.deepcopy(cur_momentum_vec)
    epsilon = step_size

    # check for infinities/rejection
    #check = calc_grad_potential_energy(q, y_pred, target)
    #if np.sum(np.isinf(check))>0:
    #    return np.inf, np.inf


    # half step update of momentum
    p -= epsilon * calc_grad_potential_energy(q, y_pred, target) / 2
    # This will use the grad of potential energy (use provided function)
    for step_id in range(n_leapfrog_steps):
        # TODO: full step update of 'position' (aka bnn_params)
        q += epsilon*p
        # This will use the grad of kinetic energy (has simple closed form)

        if step_id < (n_leapfrog_steps - 1):
            # TODO: full step update of momentum
            p -= epsilon*calc_grad_potential_energy(q, y_pred, target)
        else:
            # Special case for final step
            # TODO: half step update of momentum at the end
            p -= epsilon*calc_grad_potential_energy(q, y_pred, target)/2

    # TODO: don't forget to flip sign of momentum (ensure symmetry)
    p = -p

    prop_bnn_params = q
    prop_momentum_vec = p
    return prop_bnn_params, prop_momentum_vec



# --------------------------------------------------------------------
# NEURAL NETWORK
# --------------------------------------------------------------------

def make_bnn_architecture(input_size, hidden_units, output_size):
    # params -----------
    # initialize like this as prior is uninformative
    #with open("weights_50accuracy.pkl", "rb") as fp:
    #    parameters = pickle.load(fp)

    #w1, b1, w2, b2 = parameters
    #w1_mean = np.array(w1)
    w1_mean = np.random.normal(0, 1., (input_size, hidden_units))
    w1_std = np.random.uniform(0.,1.,(input_size, hidden_units))

    #b1_mean = np.array(b1)
    b1_mean = np.random.normal(0, 1., (hidden_units,))
    b1_std = np.random.uniform(0.,1.,(hidden_units,))

    #w2_mean = np.array(w2)
    w2_mean = np.random.normal(0, 1., (hidden_units, output_size))
    w2_std = np.random.uniform(0.,1.,(hidden_units, output_size))

    #b2_mean = np.array(b2)
    b2_mean = np.random.normal(0, 1., (output_size,))
    b2_std = np.random.uniform(0.,1.,(output_size,))


    params = [(w1_mean, w1_std), (b1_mean, b1_std), (w2_mean, w2_std), (b2_mean, b2_std)]

    # architecture --------
    L1 = layers.InputLayer(X)
    L2 = layers.FullyConnectedLayer(input_size,hidden_units)
    L3 = layers.ReLuLayer()
    L4 = layers.FullyConnectedLayer(hidden_units, output_size)
    L5 = layers.LinearLayer()
    L6 = layers.SquaredError()

    layerz = [L1, L2, L3, L4, L5, L6]

    # update with new weights n everything
    update_layerz(layerz, w1_mean, b1_mean, w2_mean, b2_mean)

    return layerz, params


def train_BNN():
    eta = 1e-2
    epsilon = 1e-10
    epochs = 100
    train_accuracies = []
    validation_accuracies = []
    input_size = X.shape[1]
    hidden_size = 10
    output_size = Y.shape[1]
    acceptance = []

    layerz, weights = make_bnn_architecture(input_size, hidden_size, output_size)
    # Training loop
    for i in tqdm(range(epochs)): # every epoch is a sample
        # Sample weights using HMC
        y_pred = forward(layerz, X)
        squashed_weights = squash_params(weights)
        samples, accept_rate = run_HMC_sampler(squashed_weights, y_pred, Y,
                                         calc_potential_energy=calc_potential_energy,
                                         calc_kinetic_energy=calc_kinetic_energy,
                                         calc_grad_potential_energy=calc_grad_potential_energy)

        # Update weights with the sampled values
        (w1, _), (b1, _), (w2, _), (b2, _) = unsquash_params(samples[-1], input_size, hidden_size, output_size)


        # Evaluate the accuracy on the test set
        layerz = update_layerz(layerz, w1, b1, w2, b2)
        validation = np.mean(np.argmax(forward(layerz, validate_X), axis=1) == np.argmax(validate_Y, axis=1))
        validation_accuracies.append(validation)
        train = np.mean(np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1))
        train_accuracies.append(train)

        acceptance.append(accept_rate)



    (w1, sw1), (b1, sb1), (w2, sw2), (b2, sb2) = unsquash_params(samples[-1], input_size, hidden_size, output_size)
    print("samples means-std of: w1, b1, w2, b2")
    print(w1.mean(), sw1.mean(), b1.mean(), sb1.mean())
    print(w2.mean(), sw2.mean(), b2.mean(), sb2.mean())

    with open("weights_bnn_boston.pkl", "wb") as fp:
        pickle.dump([(w1, sw1), (b1, sb1), (w2, sw2), (b2, sb2)] , fp)
    return train_accuracies, validation_accuracies, layerz



def forward(layerz, X):
    """ to obtain prediction from model"""
    h = X
    #"forward"
    for j in range(len(layerz)-1):
        h = layerz[j].forward(h)

    return h


def update_layerz(layerz, w1, b1, w2, b2):
    """ update layers(' FCs) with new weights"""
    layerz[1].setWeights(w1)
    layerz[1].setBiases(b1)
    layerz[3].setWeights(w2)
    layerz[3].setBiases(b2)
    return layerz




# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__=="__main__":
    train_acc, validate_acc, layerz = train_BNN()
    plot_accuracies(train_acc, validate_acc)
