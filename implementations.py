# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import numpy as np
from helpers import *
import logging as log

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_polynomial_features(x, degree):
    """ builds polynomial features"""
    phi = np.zeros((len(x), degree + 1))
    for i in range(degree+1):
        phi[:, i] = x**i 
    return phi


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    x = np.random.permutation(x)
    np.random.seed(seed)
    y = np.random.permutation(y)

    ind = int(len(x) * ratio)

    train_x = x[ : ind]
    test_x = x[ind : ]

    train_y = y[ : ind]
    test_y = y[ind : ]

    return train_x, test_x, train_y, test_y


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_score(model, x , y, loss, cv, seed):
    """
    Computes k-fold cross validation losses for a given model

    WARNING: a model should return the optimal weights and loss: w, loss
    it should have the parametres y and x for the observed values and input variables
    if a model has regularization parameters we should create a wrapper before passing, e. g.

   
    from functools import partial \\
    model = partial(ridge_regression, lambda_=1))

    Parameters:
    ----------
    model : function
        A model function which takes x and y as input and returns w, loss
    x: np.ndarray
        An N x D matrix, where N is a number of predictor variables and D is the number of featres
    y: np.ndarray
        Vector of N observed values
    loss: function
        A loss function which takes x and y as input and returns a non-negative floating point
    cv: float
        The number of folds to split the data
    seed: int
        Random seed used to initialize the pseudo-random number generator

    Returns:
    ----------
    train_scores : np.ndarray
        A vector that contains the train losses
    test_scores : np.ndarray
        A vector that contains the test losses
    """

    k_indices = build_k_indices(y, cv, seed)
    train_scores = np.zeros(cv)
    test_scores = np.zeros(cv)

    for fold in range(cv):
        x_test = x[k_indices[fold]] 
        y_test = y[k_indices[fold]]

        selector = [x for x in range(y.shape[0]) if x not in set(k_indices[fold])]
        y_train = y[selector]
        x_train = x[selector]

        w, train_loss = model(y_train,x_train)
        train_scores[fold] = train_loss
        test_scores[fold] = loss(y_test, x_test, w)

    return train_scores, test_scores

"""LINEAR REGRESSION"""

def compute_mse(y, tx, w):
    """Mean squared error"""
    e = y - tx @ w
    return 1/(2 * len(y)) * e @ e 

def compute_least_squares_gradient(y, tx, w):
    """Gradient of the MSE loss function with respect to the weights"""
    return -1 / len(y) * tx.T @ (y - tx @ w)


def least_squares_gd(y, tx, initial_w, max_iters, gamma, num_batches=1):
    """
    Linear regression model using gradient descent or stochastic gradient descent

    Parameters:
    ----------
    y : np.ndarray
        Vector of N observed values
    tx: np.ndarray
        An N x D matrix, where N is a number of predictor variables and D is the number of features
    initial_w : np.ndarray
        The initial vector of D model weights
    gamma: float
        Learning rate 
    max_iters: float, default=1
        The maximum number of iterations over the training data (aka epochs)
    num_batches: float
        The number of batches for SGD, by defaut it is set to one, resulting simple GD

    Returns:
    ----------
    w : np.ndarray
        Vector of D model weights 
    loss: 
        A non-negative floating point
    
    """
    w = initial_w
    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, 1, num_batches = num_batches):
            gradient = compute_least_squares_gradient(batch_y, batch_tx, w)
            w = w - gamma * gradient
    loss = compute_mse(y, tx ,w)
    return w, loss

def least_squares_sgd(y, tx, initial_w, max_iters, gamma): 
    """Wrapper function for Linear regression using stochastic gradient descent with batch size=1"""
    return least_squares_gd(y, tx, initial_w, max_iters, gamma, num_batches=len(y))
    
def least_squares(y, tx):
    """
    Linear regression model using normal equation 

    Parameters:
    ----------
    y : np.ndarray
        Vector of N observed values
    tx: np.ndarray
        An N x D matrix, where N is a number of predictor variables and D is the number of features

    Returns:
    ----------
    w : np.ndarray
        Vector of D model weights 
    loss: 
        A non-negative floating point
    
    """
    w = np.linalg.lstsq(tx.T @ tx, tx.T @ y)[0]
    e = y - tx @ w 
    loss = 1/len(e) * e @ e
    return w , loss
    

"""RIDGE REGRESSION"""


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression model using normal equations

    Parameters:
    ----------
    y : np.ndarray
        Vector of N observed values
    tx: np.ndarray
        An N x D matrix, where N is a number of predictor variables and D is the number of features
    lambda_ : float
        Regularization parameter

    Returns:
    ----------
    w : np.ndarray
        Vector of D model weights 
    loss: 
        A non-negative floating point

    """
    lambda1 = 2 * len(y) * lambda_
    inv = tx.T @ tx + lambda1 * np.eye(tx.shape[1])
    w = np.linalg.lstsq(inv, tx.T @ y)[0]
    loss = compute_mse(y, tx ,w)
    return w, loss


"""LOGISTIC REGRESSION"""


def sigmoid(t):
    """Apply the sigmoid function on t"""
    return 1/(1 + np.exp(-t))

def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of cross entropy loss with respect to the weights"""
    mu = sigmoid(tx @ w)
    return tx.T @ (mu - y)

def compute_cross_entropy_loss(y, tx, w):
    """Compute the negative log likelihood for logistic regression"""
    mu = sigmoid(tx @ w)
    N = len(y)
    return - (1/N) *  y.T @ np.log(mu) - (np.ones(mu.shape) - y).T @ np.log(np.ones(mu.shape) - mu)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression model using gradient descent

    Parameters:
    ----------
    y : np.ndarray
        Vector of N observed values
    tx: np.ndarray
        An N x D matrix, where N is a number of predictor variables and D is the number of features
    initial_w : np.ndarray
        The initial vector of D model weights
    max_iters: float, default=1
        The maximum number of iterations over the training data (aka epochs)
    gamma: float
        Learning rate 

    Returns:
    ----------
    w : np.ndarray
        Vector of D model weights 
    loss: float
        A non-negative floating point

    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y ,tx ,w)
        w = w - gamma * gradient
    loss = compute_cross_entropy_loss(y, tx, w)
    return w, loss

def compute_logistic_hessian(tx, w):
    """Return the Hessian of the loss function"""
    mu = sigmoid(tx @ w)
    diag = np.multiply(mu, 1-mu).T
    S = np.diag(diag.T)
    return tx.T @ S @ tx

def logistic_regression_newton_method(y, tx, initial_w, max_iters, batch_size, ratio, gamma):
    """
    Logistic regression model using Newton's method

    Parameters:
    ----------
    y : np.ndarray
        Vector of N observed values
    tx: np.ndarray
        An N x D matrix, where N is a number of predictor variables and D is the number of features
    initial_w : np.ndarray
        The initial vector of D model weights
    max_iters: float, default=1
        The maximum number of iterations over the training data (aka epochs)
    batch_size: float
        The number of training examples utilized in one iteration.
    ratio: float
        The train:test ratio
    gamma: float
        Learning rate 

    Returns:
    ----------
    w : np.ndarray
        Vector of D model weights 
    loss: float
        A non-negative floating point
    """
    train_x, test_x, train_y, test_y = split_data(tx, y, ratio, seed=1)
    num_batches = int(len(y)/batch_size)

    w = initial_w
    for n_iter in range(max_iters):
        losses = []
        for batch_y, batch_tx in batch_iter(train_y, train_x, batch_size, num_batches = num_batches):
            gradient = compute_logistic_gradient(batch_y, batch_tx, w)
            hessian = compute_logistic_hessian(batch_tx, w)
            update_vector = np.linalg.lstsq(hessian, gradient)[0]
            w = w - gamma *  update_vector
            losses.append(compute_cross_entropy_loss(train_y, train_x, w))
            
        train_loss = sum(losses) / (num_batches + 1)
        if ratio < 1:
            test_loss = compute_cross_entropy_loss(test_y, test_x, w)
            y_pred = predict_logistic_labels(w, test_x)
            test_accuracy = model_accuracy(y_pred, test_y)
            log.info(f'epoch: {n_iter}, train_loss: {train_loss}, test_loss: {test_loss}, test accuracy: {test_accuracy}')

    loss = compute_cross_entropy_loss(y, tx, w)
    return w, loss

def predict_logistic_labels(w, tx):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(tx @ w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

def model_accuracy(y_pred, y_true):
    return (1 - (sum(np.abs(y_pred-y_true)))/len(y_true))

def compute_reg_cross_entropy_loss(y, tx, w, lambda_):
    """Compute loss function for regularized logistic regression"""
    return compute_cross_entropy_loss(y, tx, w) + lambda_ / 2 * w @ w

def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma, batch_size, ratio):
    """
    Regularized logistic regression using Newton method
    
    Parameters:
    ----------
    y : np.ndarray
        Vector of N observed values
    tx: np.ndarray
        An N x D matrix, where N is a number of predictor variables and D is the number of featres
    initial_w : np.ndarray
        The initial vector of D model weights
    max_iters: float, default=1
        The maximum number of iterations over the training data (aka epochs)
    gamma: float
        Learning rate 

    Returns:
    ----------
    w : np.ndarray
        Vector of D model weights 
    loss: float
        A non-negative floating point
    """
    w = initial_w
    train_x, test_x, train_y, test_y = split_data(tx, y, ratio, seed=1)
    num_batches = int(len(y)/batch_size)

    for n_iter in range(max_iters):
        losses = []
        for batch_y, batch_tx in batch_iter(train_y, train_x, batch_size, num_batches = num_batches):

            gradient = compute_logistic_gradient(batch_y, batch_tx, w)
            regularized_gradient = gradient + 2 * lambda_ * w
            hessian = compute_logistic_hessian(batch_tx, w)
            regularized_hessian = hessian + 2 * lambda_ * np.eye(len(hessian))
            update_vector = np.linalg.lstsq(regularized_hessian, regularized_gradient)[0]
            w = w - gamma *  update_vector

            losses.append(compute_reg_cross_entropy_loss(train_y, train_x, w, lambda_))
            
        if ratio < 1:
            train_loss = sum(losses) / (num_batches + 1)
            test_loss = compute_reg_cross_entropy_loss(test_y, test_x, w, lambda_)
            y_pred = predict_logistic_labels(w, test_x)
            test_accuracy = model_accuracy(y_pred, test_y)
            log.info(f'epoch: {n_iter}, train_loss: {train_loss}, test_loss: {test_loss}, test accuracy: {test_accuracy}')

    loss = compute_reg_cross_entropy_loss(y, tx, w, lambda_)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Wrapper function for Regularized logistic regression using gradient descent or SGD"""
    return reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, batch_size=len(y), ratio=1, gamma=gamma)




