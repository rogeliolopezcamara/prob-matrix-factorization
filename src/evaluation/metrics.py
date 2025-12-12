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

def macro_mae(y_true, y_pred):
    """
    Compute Macro-Averaged Mean Absolute Error.
    MAE is computed for each unique true rating value, then averaged.
    This gives equal weight to each rating class, penalizing models that
    fail to predict rare classes (e.g., ratings of 1, 2, 3 in a skewed dataset).
    """
    labels = np.unique(y_true)
    maes = []
    for label in labels:
        mask = (y_true == label)
        if np.any(mask):
            mae_k = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            maes.append(mae_k)
    return np.mean(maes)

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