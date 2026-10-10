"""
logistic.py  –  Task 3 helpers: logistic regression from scratch.

Implements:
  - sigmoid(z)
  - hypothesis(X, theta)
  - cost(X, y, theta)
  - fit_logistic(X, y, alpha, n_iters)
  - predict(X, theta)

No sklearn is used; only NumPy.
"""

import numpy as np


def sigmoid(z):
    """Logistic (sigmoid) function: g(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def hypothesis(X, theta):
    """
    Compute h_theta(X) = sigmoid(X @ theta).

    Parameters
    ----------
    X : ndarray, shape (m, d+1)   – feature matrix with bias column x0=1
    theta : ndarray, shape (d+1,) – parameter vector

    Returns
    -------
    ndarray, shape (m,) – predicted probabilities
    """
    return sigmoid(X @ theta)


def cost(X, y, theta):
    """
    Negative log-likelihood cost  J(theta) = -(1/n) * ell(theta).

    Clips h to [1e-12, 1-1e-12] before taking the log to avoid log(0).
    """
    m = len(y)
    h = hypothesis(X, theta)
    h = np.clip(h, 1e-12, 1.0 - 1e-12)
    J = -(1.0 / m) * (y @ np.log(h) + (1.0 - y) @ np.log(1.0 - h))
    return J


def fit_logistic(X, y, alpha, n_iters):
    """
    Batch gradient descent for logistic regression.

    The update rule (same form as the LMS rule):
        theta_j := theta_j + (alpha / n) * sum_i (y_i - h(x_i)) * x_ij

    Parameters
    ----------
    X : ndarray, shape (m, d+1)
    y : ndarray, shape (m,)
    alpha : float – learning rate
    n_iters : int – number of iterations

    Returns
    -------
    theta : ndarray, shape (d+1,)
    cost_history : list of float – J(theta) after every iteration
    """
    m, d = X.shape
    theta = np.zeros(d)
    cost_history = []

    for _ in range(n_iters):
        h = hypothesis(X, theta)
        gradient = (1.0 / m) * (X.T @ (y - h))   # shape (d,)
        theta = theta + alpha * gradient
        cost_history.append(cost(X, y, theta))

    return theta, cost_history


def predict(X, theta):
    """Return 1 where h_theta(x) >= 0.5, else 0."""
    probs = hypothesis(X, theta)
    return (probs >= 0.5).astype(int)
