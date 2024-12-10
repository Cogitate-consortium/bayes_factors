from scipy.special import betaln
import numpy as np
from scipy.stats import beta, binom
from scipy.stats import gamma, binomtest
import matplotlib.pyplot as plt


import numpy as np
import pingouin as pg


def bayes_ttest(x, y=0, paired=False, alternative='two-sided', r=0.707):
    """
    Compute Bayes Factors from a t-test using Pingouin's JZS method, applied to 
    data arrays that can be 1D, 2D, or 3D. 
    
    Parameters
    ----------
    x : array
        Can be of shape:
         - (n): n observations
         - (n, t): n observations at each time point
         - (n, t, freq): n observations at each time-frequency "pixel"
         - (n, t, t): n observations at each time-by-time "pixel"
    y : array or scalar, default=0
        If scalar, a one-sample t-test (x vs mu) is performed.
        If array and same shape as x (except first dim), a two-sample test is performed.
    paired : bool, default=False
        If True, perform a paired t-test (requires x and y to have identical shape).
    alternative : str, default='two-sided'
        Defines the alternative hypothesis. Must be 'two-sided', 'greater', or 'less'.
    r : float, default=0.707
        Prior width for the JZS Bayes factor computation.

    Returns
    -------
    results : dict
        A dictionary containing:
        - 'T': array of t-values
        - 'pval': array of p-values
        - 'BF10': array of Bayes factors (BF10)
        
        If x is 1D, these are scalars.
        If x is multi-dimensional (2D, 3D), these are arrays of shape 
        matching the non-observation dimensions of x.
    """
    x = np.asarray(x)
    
    # Check if y is a scalar or array
    if np.isscalar(y):
        # One-sample test scenario
        y_is_scalar = True
        y_val = float(y)
    else:
        y_is_scalar = False
        y = np.asarray(y)

    # Basic input checks
    if x.ndim < 1 or x.ndim > 3:
        raise ValueError(f"x must be 1D, 2D, or 3D, but got {x.ndim}D")

    # If two-sample, ensure shape compatibility
    if not y_is_scalar:
        if x.shape != y.shape:
            raise ValueError("For a two-sample test, x and y must have the same shape.")
    
    # Check paired requirement
    if paired and y_is_scalar:
        raise ValueError("For a paired test, y must be an array, not a scalar.")
    
    # Determine test size and print info
    # We always have n in dimension 0
    shape_without_n = x.shape[1:]
    if y_is_scalar:
        test_type = "one-sample"
    else:
        if paired:
            test_type = "paired"
        else:
            test_type = "two-sample"
    
    # Determine how many tests we run
    if x.ndim == 1:
        # single test
        n_tests = 1
        shape_of_output = ()
    else:
        # multiple tests
        shape_of_output = shape_without_n
        n_tests = np.prod(shape_without_n)
    
    print(f"We will conduct a {test_type} t-test for {n_tests} point(s).")

    # Function to run a single t-test
    def run_ttest(x_data, y_data):
        # pg.ttest requires arrays for both x and y if not testing a scalar
        # If testing against scalar, provide y_data = None and use x_data vs 0
        if y_is_scalar:
            # One-sample test against y_val
            res = pg.ttest(x_data, np.ones_like(x_data)*y_val, paired=paired, 
                           alternative=alternative, r=r)
        else:
            # Two-sample test
            res = pg.ttest(x_data, y_data, paired=paired, 
                           alternative=alternative, r=r)
        return res

    # Prepare result containers
    if n_tests == 1:
        res = pg.ttest(x, y, paired=paired, 
                alternative=alternative, r=r)
        return res['BF10'].values[0]
    else:
        # Multiple tests: loop over the shape and run test for each slice
        BF_out = np.zeros(shape_of_output)

        # We will iterate over all indices in shape_without_n using np.ndindex
        for idx in np.ndindex(shape_of_output):
            # Construct slice
            # For (n), idx = () empty
            # For (n, t), idx = (time_point,)
            # For (n, t, freq), idx = (time_point, freq_point)
            # The data slice will be x[:, idx...]
            # where idx can be used directly inside indexing
            x_slice = x[(slice(None),) + idx]
            if not y_is_scalar:
                y_slice = y[(slice(None),) + idx]
            else:
                y_slice = y

            # run the test
            res = pg.ttest(x_slice, y_slice, paired=paired, alternative=alternative, r=r)
            BF_out[idx] = res['BF10'].values[0]

        return BF_out


def sim_decoding_binomial(t0, tmax, sfreq, scale_factor=3, tstart=0, ntrials=100):

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

