from scipy.special import betaln
import numpy as np
from scipy.stats import beta, binom
from scipy.stats import gamma, binomtest
import matplotlib.pyplot as plt


def sim_decoding(t0, tmax, sfreq, scale_factor=3, tstart=0, ntrials=100):

    # Simulate a time series of decoding accuracy:
    times = np.linspace(t0, tmax, sfreq * (tmax - t0))
    # Create a time series of decoding accuracy:
    decoding_accuracy = gamma.pdf(times, 3, scale=0.2, loc=tstart)
    # Normalize it:
    decoding_accuracy_true = 0.5 + (decoding_accuracy/np.max(decoding_accuracy))/scale_factor

    # Loop through each time points:
    obs = []
    for i, t in enumerate(times):
        k = np.random.binomial(ntrials, decoding_accuracy_true[i])  # number of successes
        # Create the vector:
        y = np.array([1] * int(k) + [0] * int(ntrials - k))
        # Compute the bayes factor:
        obs.append(y)

    return obs


def beta_binom_evidence(prior, y):
    """
    Compute the marginal likelihood, analytically, for a beta-binomial model.

    prior : tuple
        tuple of alpha and beta parameter for the prior (beta distribution)
    y : array
        array with "1" and "0" corresponding to the success and fails respectively
    """
    alpha, beta = prior
    h = np.sum(y)
    n = len(y)
    p_y = np.exp(betaln(alpha + h, beta + n - h) - betaln(alpha, beta))
    return p_y


def binomial_bf(y, theta_h0=0.5, prior_h1=[1, 1]):

    # Make sure that the input is a numpy array:
    
    # Compute the probability of the data null:
    p_data_given_H0 = binom.pmf(np.sum(y), y.shape[0], theta_h0)
    # Compute the probability of the data given the alternative hypothesis:
    p_data_given_H1 = beta_binom_evidence(prior_h1, y)

    return p_data_given_H1/p_data_given_H0

