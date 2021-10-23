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

def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_least_squares_gradient(y, tx, w)
        w = w- gamma * gradient
    loss = compute_mse(y, tx ,w)
    return w, loss
    

def least_squares_sgd(y, tx, initial_w, max_iters, gamma): 
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for batch_y, batch_tx in batch_iter(y, tx, 1, num_batches = max_iters):
        grad = compute_least_squares_gradient(batch_y, batch_tx, w)
        w = w - gamma * grad
    loss = compute_mse(batch_y, batch_tx, w)
    return w, loss
    

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx @ w 
    loss = 1/len(e) * e @ e
    return w , loss
    

"""RIDGE REGRESSION"""


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    lambda1 = 2 * len(y) * lambda_
    inv = tx.T @ tx + lambda1 * np.eye(tx.shape[1])
    w = np.linalg.inv(inv) @ tx.T @ y
    loss = compute_mse(y, tx ,w)
    return w, loss


"""LOGISTIC REGRESSION"""


def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1 + np.exp(-t))

def compute_logistic_grad(y, tx, w):
    return 1 / len(y) * (tx.T @ (h_model(tx, w)-y))

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of cross entropy loss with respect to the weights"""
    mu = np.apply_along_axis(sigmoid, 0, tx @ w)
    return tx.T @ (mu - y)

def h_model(tx ,w):
    return sigmoid(-w @ tx)

def compute_cross_entropy_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    mu = np.apply_along_axis(sigmoid, 0, tx @ w)
    return - y.T @ np.log(mu) - (np.ones(mu.shape) - y).T @ np.log(np.ones(mu.shape) - mu)


def logistic_regression_gd(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_logistic_grad(y ,tx ,w)
        w = w - gamma * gradient
    loss = compute_cross_entropy_loss(y, tx, w)
    return w, loss

def compute_logistic_hessian(tx, w):
    """return the Hessian of the loss function."""
    mu = np.apply_along_axis(sigmoid, 0, tx @ w)
    diag = np.multiply(mu, 1-mu).T[0]
    S = np.diag(diag.T)
    return tx.T @ S @ tx

def logistic_regression_newton_method(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using Newton's method, return the loss and  w."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        hessian = compute_logistic_hessian(tx, w)
        update_vector = np.linalg.solve(hessian, gradient)
        w = w - gamma *  update_vector
    loss = compute_cross_entropy_loss(y, tx, w)
    return w, loss

def compute_reg_cross_entropy_loss(y, tx, w, lambda_):
    return compute_cross_entropy_loss(y, tx, w) + lambda_ / 2 * w @ w

def reg_logistic_regression_gd(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        regularized_gradient = gradient + 2 * lambda_ * w
        ##hessian = compute_logistic_hessian(y, tx, w)
        ## regularized_hessian = hessian + 2 * lambda_ * np.eye(len(hessian))
        w = w - gamma *  regularized_gradient
    loss = compute_reg_cross_entropy_loss(y, tx, w)
    return w, loss
    
