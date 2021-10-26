# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

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


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


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
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
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
    w = np.linalg.inv(inv) @ tx.T @ y
    loss = compute_mse(y, tx ,w)
    return w, loss


"""LOGISTIC REGRESSION"""


def sigmoid(t):
    """Apply the sigmoid function on t"""
    return 1/(1 + np.exp(-t))

def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of cross entropy loss with respect to the weights"""
    mu = np.apply_along_axis(sigmoid, 0, tx @ w)
    return tx.T @ (mu - y)

def compute_cross_entropy_loss(y, tx, w):
    """Compute the negative log likelihood for logistic regression"""
    mu = np.apply_along_axis(sigmoid, 0, tx @ w)
    return - y.T @ np.log(mu) - (np.ones(mu.shape) - y).T @ np.log(np.ones(mu.shape) - mu)


def logistic_regression_gd(y, tx, initial_w, max_iters, gamma):
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
    mu = np.apply_along_axis(sigmoid, 0, tx @ w)
    diag = np.multiply(mu, 1-mu).T[0]
    S = np.diag(diag.T)
    return tx.T @ S @ tx

def logistic_regression_newton_method(y, tx, initial_w, max_iters, gamma):
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
        gradient = compute_logistic_gradient(y, tx, w)
        hessian = compute_logistic_hessian(tx, w)
        update_vector = np.linalg.solve(hessian, gradient)
        w = w - gamma *  update_vector
    loss = compute_cross_entropy_loss(y, tx, w)
    return w, loss

def compute_reg_cross_entropy_loss(y, tx, w, lambda_):
    """Compute loss function for regularized logistic regression"""
    return compute_cross_entropy_loss(y, tx, w) + lambda_ / 2 * w @ w

def reg_logistic_regression_gd(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    
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
    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        regularized_gradient = gradient + 2 * lambda_ * w
        ##hessian = compute_logistic_hessian(y, tx, w)
        ## regularized_hessian = hessian + 2 * lambda_ * np.eye(len(hessian))
        w = w - gamma *  regularized_gradient
    loss = compute_reg_cross_entropy_loss(y, tx, w)
    return w, loss

def predict_logistic_regression(w, x):
    return 


""" HELPERS """

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
    scores : np.ndarray
        A cv x 2 matrix, the first column contains the train losses, the second containt the test loss for each fold
    """

    k_indices = build_k_indices(y, cv, seed)
    scores = np.zeros(cv)
    for fold in range(cv):
        x_test = x[k_indices[fold]] 
        y_test = y[k_indices[fold]]

        selector = [x for x in range(y.shape[0]) if x not in set(k_indices[fold])]
        y_train = y[selector]
        x_train = x[selector]

        w, train_loss = model(y_train,x_train)
        train_loss = loss(y_train, x_train, w)
        test_loss = loss(y_test, x_test, w)
        scores[fold] = np.array([train_loss, test_loss])

    return scores