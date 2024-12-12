from scipy.special import betaln
from scipy.special import beta as beta_function
from scipy.stats import beta, binom
from scipy.stats import gamma, binomtest
from scipy.integrate import quad

import numpy as np
import warnings
import pingouin as pg


def bayes_binomtest(k, n, p=0.5, a=1, b=1, return_pval=False):
    """
    Compute Bayes Factors from a binomial test using a Beta-Binomial model,
    applied to data that can be 0D, 1D, or 2D.
    
    Parameters
    ----------
    k : float, int, or array-like
        Observed number of successes or observed accuracy.
        If values are between 0 and 1, they will be interpreted as a probability 
        of success and converted to number of successes by multiplying by n 
        and rounding to the nearest integer.
        
        Shapes allowed:
         - 0D: a single value
         - 1D: (m) - e.g., decoding accuracies at multiple time points
         - 2D: (m, l) - e.g., decoding accuracies for time-by-time generalization matrix
         
    n : int
        Number of trials.
        
    p : float, default=0.5
        The null hypothesis probability of success.
        
    a : float, default=1
        Prior alpha parameter for the Beta distribution. Together with b, this defines
        the prior Beta(a, b).
        
    b : float, default=1
        Prior beta parameter for the Beta distribution.
        
    Returns
    -------
    BF10 : scalar, 1D array, or 2D array
        The Bayes factor for each test location. Matches the shape of the input `k`, 
        excluding the trial dimension (since `n` is a scalar).
        
    Notes
    -----
    This function is meant for applying a binomial Bayes factor test to single decoding accuracy
    values or small sets thereof (e.g. across time or across a time-by-time matrix).
    It is NOT intended for computing Bayes factors across subjects. For example, you might have
    already averaged decoding accuracies across subjects and wish to test if the decoding accuracy
    is better than chance.
    """
    # Convert input to array for consistent indexing
    k = np.asarray(k)
    
    # Determine the dimensionality
    if k.ndim > 2:
        raise ValueError("k must be 0D, 1D, or 2D.")
    
    # If k is a probability (between 0 and 1), convert to count of successes
    # We'll check if all values are between 0 and 1 or not
    if np.issubdtype(k.dtype, np.floating):
        # Check if all values are in [0,1]
        if np.all((k >= 0) & (k <= 1)):
            # Convert to integer successes
            successes = k * n
            # Check if close to integer
            if not np.allclose(successes, np.round(successes), atol=1e-7):
                warnings.warn(
                    "Some values of k*n are not integers. Rounding to nearest integer.",
                    UserWarning
                )
            successes = np.round(successes).astype(int)
        else:
            # If not all in [0,1], assume already raw counts
            raise ValueError("Values must be int or floats between 0 and 1!")
    else:
        # If k is integer, assume already counts
        successes = k.astype(int)
    
    # Determine the shape for the output
    # If k is 0D, shape_of_output = ()
    # If 1D, shape_of_output = (m,)
    # If 2D, shape_of_output = (m, l)
    shape_of_output = k.shape
    
    # Determine how many tests
    if k.ndim == 0:
        n_tests = 1
    else:
        n_tests = np.prod(shape_of_output)
    
    print(f"Conducting {n_tests} binomial Bayes factor test(s).")
    
    # If only one test (0D)
    if k.ndim == 0:
        # Single test
        bf = pg.bayesfactor_binom(int(successes), n, p=p, a=a, b=b)
        if return_pval:
            res_freq = binomtest(int(successes), n, p=p)
            return bf, res_freq.pvalue
        else:
            return bf
    
    elif k.ndim == 1:
        # Multiple tests along one dimension
        BF_out = np.zeros(shape_of_output)
        if return_pval:
            pval_out = np.zeros(shape_of_output)
        for i in range(shape_of_output[0]):
            res = pg.bayesfactor_binom(int(successes[i]), n, p=p, a=a, b=b)
            BF_out[i] = res
            if return_pval:
                pval_out[i] = binomtest(int(successes[i]), n, p=p).pvalue
        if return_pval:
            return BF_out, pval_out
        else:
            return BF_out
    
    else:
        # k.ndim == 2
        BF_out = np.zeros(shape_of_output)
        if return_pval:
            pval_out = np.zeros(shape_of_output)
        for idx in np.ndindex(shape_of_output):
            res = pg.bayesfactor_binom(int(successes[idx]), n, p=p, a=a, b=b)
            BF_out[idx] = res
            if return_pval:
                pval_out[idx] = binomtest(int(successes[idx]), n, p=p).pvalue
        if return_pval:
            return BF_out, pval_out
        else:
            return BF_out



def bayes_ttest(x, y=0, paired=False, alternative='two-sided', r=0.707, return_pval=False):
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

    # Prepare result containers
    if n_tests == 1:
        res = pg.ttest(x, y, paired=paired, 
                alternative=alternative, r=r)
        if return_pval:
            return res['BF10'].values[0], res['p-val'].values[0]
        else:
            return res['BF10'].values[0]
    else:
        # Multiple tests: loop over the shape and run test for each slice
        BF_out = np.zeros(shape_of_output)
        if return_pval:
            pval_out = np.zeros(shape_of_output)
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
            if return_pval:
                pval_out[idx] = res['p-val'].values[0]
        if return_pval:
            return BF_out, pval_out
        else:
            return BF_out


def sim_decoding_binomial(t0, tmax, sfreq, scale_factor=3, tstart=0, ntrials=100):
    """
    Simulate a 1D time series of decoding accuracy using a binomial distribution.

    Parameters
    ----------
    t0 : int
        Start time of the simulation (in seconds).
    tmax : int
        End time of the simulation (in seconds).
    sfreq : int
        Sampling frequency (number of samples per second).
    scale_factor : float, optional
        Factor to normalize the decoding accuracy. Default is 3.
    tstart : float, optional
        Location parameter for the gamma distribution. Default is 0.
    ntrials : int, optional
        Number of trials for the binomial distribution. Default is 100.

    Returns
    -------
    numpy.ndarray
        Simulated number of successes (k) for each time point in the time series.
    """
    # Simulate a time series of decoding accuracy:
    times = np.linspace(t0, tmax, sfreq * (tmax - t0))
    # Create a time series of decoding accuracy:
    decoding_accuracy = gamma.pdf(times, 3, scale=0.2, loc=tstart)
    # Normalize it:
    decoding_accuracy_true = 0.5 + (decoding_accuracy/np.max(decoding_accuracy))/scale_factor

    # Loop through each time points:
    k = np.random.binomial(ntrials, decoding_accuracy_true)  # number of successes

    return k



def sim_decoding_binomial_2d(t0, tmax, sfreq, scale_factor=3, tstart=0, ntrials=100):
    """
    Simulate a 2D matrix of decoding accuracy using a binomial distribution.

    Parameters
    ----------
    t0 : int
        Start time of the simulation (in seconds).
    tmax : int
        End time of the simulation (in seconds).
    sfreq : int
        Sampling frequency (number of samples per second).
    scale_factor : float, optional
        Factor to normalize the decoding accuracy matrix. Default is 3.
    tstart : float, optional
        Location parameter for the gamma distribution. Default is 0.
    ntrials : int, optional
        Number of trials for the binomial distribution. Default is 100.

    Returns
    -------
    numpy.ndarray
        Simulated 2D array of binomial observations for decoding accuracy.
    """
    # Simulate a time series of decoding accuracy:
    times = np.linspace(t0, tmax, sfreq * (tmax - t0))
    # Create a time series of decoding accuracy:
    decoding_accuracy = gamma.pdf(times, 3, scale=0.2, loc=tstart)
    # Calcuate the probability matrix:
    prob_matrix = np.outer(decoding_accuracy, decoding_accuracy)

    # Normalize it:
    prob_matrix_norm = 0.5 + (prob_matrix/np.max(prob_matrix))/scale_factor

    # Create observation
    obs = np.random.binomial(n=ntrials, p=prob_matrix_norm)
    return obs

from scipy.special import beta as beta_func
from scipy.stats import norm


def scaled_beta_tau(tau, alpha=1.0, beta=1.0):
    """
    Compute the scaled Beta prior for Kendall's tau.

    Parameters
    ----------
    tau : float or ndarray
        The Kendall's tau value(s) at which to evaluate the prior.
    alpha : float, optional
        The alpha parameter for the Beta function, by default 1.0.
    beta : float, optional
        The beta parameter for the Beta function, by default 1.0.

    Returns
    -------
    float or ndarray
        The value(s) of the scaled beta prior at tau.
    """
    tau = np.atleast_1d(tau)
    val = ((np.pi * 2 ** (-2 * alpha)) / beta_func(alpha, alpha)) * (
        np.cos((np.pi * tau) / 2) ** (2 * alpha - 1)
    )
    return val if val.size > 1 else val[0]


def prior_tau(tau, kappa=1.0):
    """
    Compute the prior distribution p(tau) for two-sided tests.

    p(tau) = scaledBetaTau(tau, alpha=1/kappa, beta=1/kappa)

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    kappa : float, optional
        The parameter that controls the prior shape, by default 1.0.

    Returns
    -------
    float or ndarray
        Prior density at tau.
    """
    return scaled_beta_tau(tau, alpha=1.0 / kappa, beta=1.0 / kappa)


def prior_tau_positive(tau, kappa=1.0):
    """
    Compute the one-sided prior distribution p(tau) for tau in [0,1].

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    kappa : float, optional
        The parameter controlling the shape of the prior, by default 1.0.

    Returns
    -------
    float or ndarray
        Prior density at tau (0 if tau outside [0,1]).
    """
    tau = np.atleast_1d(tau)
    result = np.zeros_like(tau)
    idx = (tau >= 0) & (tau <= 1)
    result[idx] = 2 * prior_tau(tau[idx], kappa)
    return result if result.size > 1 else result[0]


def prior_tau_negative(tau, kappa=1.0):
    """
    Compute the one-sided prior distribution p(tau) for tau in [-1,0].

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    kappa : float, optional
        The parameter controlling the shape of the prior, by default 1.0.

    Returns
    -------
    float or ndarray
        Prior density at tau (0 if tau outside [-1,0]).
    """
    tau = np.atleast_1d(tau)
    result = np.zeros_like(tau)
    idx = (tau >= -1) & (tau <= 0)
    result[idx] = 2 * prior_tau(tau[idx], kappa)
    return result if result.size > 1 else result[0]


def post_density_kendall_tau(tau, T_star, n, kappa=1.0, var=1.0, test="two-sided"):
    """
    Compute the unnormalized posterior density for Kendall's tau.

    The posterior is proportional to p(T_star|tau)*p(tau).

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s).
    T_star : float
        Standardized Kendall's tau statistic.
    n : int
        Sample size.
    kappa : float, optional
        Parameter controlling the prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood. min(var,1) is used, by default 1.0.
    test : str, optional
        Type of test: "two-sided", "positive", or "negative". By default "two-sided".

    Returns
    -------
    float or ndarray
        Unnormalized posterior density at tau.
    """
    tau = np.atleast_1d(tau)
    var = min(1.0, var)

    # Select prior function based on test type
    if test == "two-sided":
        prior_func = prior_tau
    elif test == "positive":
        prior_func = prior_tau_positive
    elif test == "negative":
        prior_func = prior_tau_negative
    else:
        raise ValueError("test must be 'two-sided', 'positive', or 'negative'.")

    # Likelihood for T* given tau
    likelihood = norm.pdf(T_star, loc=1.5 * tau * np.sqrt(n), scale=np.sqrt(var))
    posterior_unnormalized = likelihood * prior_func(tau)

    return posterior_unnormalized if posterior_unnormalized.size > 1 else posterior_unnormalized[0]


def posterior_tau(tau, kentau, n, kappa=1.0, var=1.0, test="two-sided"):
    """
    Compute the normalized posterior density p(tau|data).

    Parameters
    ----------
    tau : float or ndarray
        Kendall's tau value(s) at which to evaluate the posterior.
    kentau : float
        Observed Kendall's tau.
    n : int
        Sample size.
    kappa : float, optional
        Parameter controlling prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood (min(1,var) used), by default 1.0.
    test : str, optional
        Type of test: "two-sided", "positive", or "negative", by default "two-sided".

    Returns
    -------
    float or ndarray
        The posterior density at tau.
    """
    tau = np.atleast_1d(tau)
    var = min(1.0, var)

    # Compute T*
    T_star = (kentau * ((n * (n - 1)) / 2.0)) / np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)

    # Integration limits based on test
    if test == "two-sided":
        lims = (-1, 1)
    elif test == "positive":
        lims = (0, 1)
    elif test == "negative":
        lims = (-1, 0)
    else:
        raise ValueError("test must be 'two-sided', 'positive', or 'negative'.")

    def integrand(x):
        return post_density_kendall_tau(x, T_star, n, kappa, var, test=test)

    integral_val, _ = quad(integrand, lims[0], lims[1])

    post_vals = post_density_kendall_tau(tau, T_star, n, kappa, var, test=test) / integral_val
    return post_vals


def bf_kendall_tau(tau, n, kappa=1.0, var=1.0):
    """
    Compute Bayes factors for Kendall's tau.

    Parameters
    ----------
    tau : float
        Observed Kendall's tau.
    n : int
        Sample size.
    kappa : float, optional
        Parameter controlling the prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood, by default 1.0.

    Returns
    -------
    dict
        A dictionary with keys 'n', 'r', 'bf10', 'bfPlus0', and 'bfMin0' representing sample size,
        observed tau, and the three Bayes factors respectively.
    """
    result = {
        'n': n,
        'r': tau,
        'bf10': None,
        'bfPlus0': None,
        'bfMin0': None
    }

    # BF10 (two-sided)
    result['bf10'] = prior_tau(0, kappa) / posterior_tau(0, tau, n, kappa=kappa, var=var, test="two-sided")

    # BF+0 (one-sided, positive)
    result['bfPlus0'] = prior_tau_positive(0, kappa) / posterior_tau(0, tau, n, kappa=kappa, var=var, test="positive")

    # BF-0 (one-sided, negative)
    result['bfMin0'] = prior_tau_negative(0, kappa) / posterior_tau(0, tau, n, kappa=kappa, var=var, test="negative")

    return result


def kendall_bf_from_data(x, y, kappa=1.0, var=1.0):
    """
    Compute Bayes factors for Kendall's tau given raw data.

    Parameters
    ----------
    x : array_like
        Data vector for variable X.
    y : array_like
        Data vector for variable Y.
    kappa : float, optional
        Parameter controlling the prior shape, by default 1.0.
    var : float, optional
        Variance parameter for the likelihood, by default 1.0.

    Returns
    -------
    tuple
        (tau, bf_result) where tau is the Kendall's tau and bf_result is a dictionary of Bayes factors.
    """
    res = pg.corr(x, y, method='kendall')
    tau_val = res['r'].values[0]
    n = len(x)
    bf_result = bf_kendall_tau(tau_val, n, kappa=kappa, var=var)
    return tau_val, res['p-val'].values[0], bf_result


if __name__ == "__main__":
    # Example usage
    yourKendallTauValue = -0.3
    yourN = 20

    # Compute BFs with given tau and N
    bf_result = bf_kendall_tau(yourKendallTauValue, yourN)
    print("Bayes Factors:", bf_result)
