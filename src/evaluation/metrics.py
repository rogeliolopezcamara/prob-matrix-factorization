# src/evaluation/metrics.py

import numpy as np
from scipy.special import gammaln

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error between true and predicted ratings.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

def GaussianLogPredictiveLikelihood(df, theta, beta, sigma):
    """
    Compute Gaussian Log Predictive Likelihood.
    """
    # calculate expectation
    predictions = np.sum(theta[df.u] * beta[df.i], axis=1)
    
    # residuals
    squared_errors = (df.rating - predictions) ** 2
    
    # auxiliar quantities for log likelihood
    n_samples = df.shape[0]
    variance = sigma ** 2
    
    # log likelihood
    total_log_likelihood = np.sum(-0.5*np.log(2*np.pi*variance)-squared_errors/(2*variance))

    return total_log_likelihood

def PoissonLogPredictiveLikelihood(df,theta,beta,epsilon=1e-10):
    """
    Compute Poission Log Predictive Likelihood.
    """
    # calculate expectation
    lambdas = np.sum(theta[df.u] * beta[df.i], axis=1)
    
    # add epsilon if lambda=0 to avoid log(0)
    lambdas = np.maximum(lambdas, epsilon)
    
    # log likelihood
    total_log_likelihood=np.sum(df.rating*np.log(lambdas)-lambdas-gammaln(df.rating+1))
    
    return total_log_likelihood