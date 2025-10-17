#this file performs different types of regression and returns the estimated coefficients
import numpy as np

def ridgeless_beta_hat(X, y):
    """
    Run ridgeless regression using manual implementation of least squares formulas.
    
    This function implements ridgeless (ordinary) least squares regression by solving
    the normal equations. It automatically handles both underdetermined (p >= n) and
    overdetermined (p < n) cases using the appropriate mathematical formulas.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix. An intercept column will be automatically added.
    y : array-like of shape (n_samples,)
        Target values (dependent variable).
    
    Returns
    -------
    beta_hat : ndarray of shape (n_features + 1,)
        Estimated regression coefficients including intercept.
        beta[0] is the intercept, beta[1:] are the feature coefficients.
    
    Raises
    ------
    ValueError
        If X is not 2-dimensional or y is not 1-dimensional.
    
    Notes
    -----
    The function uses two different formulas depending on the problem dimensions:
    - When p < n: β̂ = (X^T X)^(-1) X^T y (overdetermined case)
    - When p ≥ n: β̂ = X^T (XX^T)^(-1) y (underdetermined case)
    
    For numerical stability, np.linalg.solve() is used instead of matrix inversion.
    """
    # Manual implementation using the formulas
    n, p = X.shape

    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("Invalid input shapes")
    
    
    if p <= n:
        # when p < n: β̂ = (X^T X)^(-1) X^T y
        XTX = X.T @ X
        XTy = X.T @ y
        beta_hat = np.linalg.solve(XTX, XTy)  # More stable than inv(XTX) @ XTy
    else:
        # when p ≥ n: β̂ = X^T (XX^T)^(-1) y  
        XXT = X @ X.T
        beta_hat = X.T @ np.linalg.solve(XXT, y)  # More stable than X.T @ inv(XXT) @ y
    
    return beta_hat

def calculate_mse(beta_hat, beta_true, gamma, sigma_squared=1, r_squared=5):
    """
    Calculate the mean squared error (MSE) of the ridgeless regression estimator.
    
    Parameters
    ----------
    beta_hat : array-like
        Estimated regression coefficients.
    beta_true : array-like
        True regression coefficients.
    sigma_squared : float
        Noise variance.
    gamma : float
        Aspect ratio (p/n).
    
    Returns
    -------
    mse : float
        Mean squared error of the estimator.
    
    Notes
    -----
    Uses the theoretical MSE formula for ridgeless regression:
    - When γ < 1: MSE = σ² * γ/(1-γ)
    - When γ > 1: MSE = r²(1 - 1/γ) + σ² * 1/(γ-1)
    """
    experimental_mse = np.mean((beta_hat - beta_true)**2)
    
    theoretical_mse = None
    if gamma < 1:
        # When γ < 1 (overdetermined case)
        theoretical_mse = sigma_squared * gamma / (1 - gamma)
    elif gamma > 1:
        # When γ > 1 (underdetermined case)
        theoretical_mse = r_squared * (1 - 1/gamma) + sigma_squared * 1 / (gamma - 1)
    else:
        # When γ = 1 (critical case)
        raise ValueError("Gamma = 1 is a critical case where MSE is undefined")
    
    return experimental_mse, theoretical_mse


# from scipy import stats
# import warnings
# from sklearn.linear_model import HuberRegressor
# from sklearn.linear_model import QuantileRegressor
# from sklearn.linear_model import LinearRegression
# 
# def lin_regression(X, y):
#     """Linear regression using sklearn"""
# 
#     if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
#         raise TypeError("Both X and y must be arrays or lists")
#     # check if X is full rank
#     if np.linalg.matrix_rank(X) < min(X.shape):
#         warnings.warn("X is not full rank. Results may be unreliable.", UserWarning)
# 
#     model = LinearRegression()
#     model.fit(X, y)
#     return model.coef_
# 
# def huber_regression(X, y, epsilon=1.35):
#     """Huber regression using sklearn"""
# 
#     if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
#         raise TypeError("Both X and y must be arrays or lists")
#     
#     model = HuberRegressor(epsilon=epsilon, max_iter=100000)
#     model.fit(X, y)
#     return model.coef_
# 
# def quantile_regression(X, y, alpha =0.5):
#     """Quantile regression using sklearn"""
# 
#     if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
#         raise TypeError("Both X and y must be arrays or lists")
#     
#     model = QuantileRegressor(quantile=alpha, solver='highs')
#     model.fit(X, y)
#     return model.coef_


