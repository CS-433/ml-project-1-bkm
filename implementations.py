# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt

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
    w = np.linalg.inv(inv) @ tx.T @ y
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
            print(f'epoch: {n_iter}, train_loss: {train_loss}, test_loss: {test_loss}, test accuracy: {test_accuracy}')

    loss = compute_cross_entropy_loss(y, tx, w)
    return w, loss

def predict_logistic_labels(w, tx):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(tx @ w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred

def model_accuracy(y_pred, y_true):
    return 1 - (sum(np.abs(y_pred-y_true))/len(y_true))


def compute_reg_cross_entropy_loss(y, tx, w, lambda_):
    """Compute loss function for regularized logistic regression"""
    return compute_cross_entropy_loss(y, tx, w) + lambda_ / 2 * w @ w

def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, batch_size, ratio, gamma):
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

        train_loss = sum(losses) / (num_batches + 1)
        test_loss = compute_reg_cross_entropy_loss(test_y, test_x, w, lambda_)
        y_pred = predict_logistic_labels(w, test_x)
        test_accuracy = model_accuracy(y_pred, test_y)
        print(f'epoch: {n_iter}, train_loss: {train_loss}, test_loss: {test_loss}, test accuracy: {test_accuracy}')

    loss = compute_reg_cross_entropy_loss(y, tx, w, lambda_)
    return w, loss


"""VISUALIZATION"""

def plot_correlation_heatmap(tx):

    features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']

    corr = np.corrcoef(tx, rowvar=False)


    fig, ax = plt.subplots(figsize=(25,25))
    im = ax.imshow(corr,  cmap='viridis')
    ax.set_xticks(np.arange(30))
    ax.set_yticks(np.arange(30))
    ax.set_xticklabels(features)
    ax.set_yticklabels(features)


    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

    for i in range(30):
        for j in range(30):
            text = ax.text(j, i, "{:.2f}".format(corr[i, j]),ha="center", va="center", color="w")

    ax.set_title("Feature correlations")
    fig.tight_layout()
    plt.show()

def plot_feature_distribution(tx, y, feature, bins):

    features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
    
    idx = np.arange(30)
    feature_by_index = {features[i]: idx[i] for i in range(30)}

    data = tx.T[feature_by_index[feature]]
    
    data_1 = data[np.where(y == 1)]
    data_0 = data[np.where(y == 0)]

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(data_1, bins=bins, color='red', alpha=0.5, label='1')
    ax.hist(data_0, bins=bins, color='blue',alpha=0.5, label='0')
    ax.legend()
    ax.set_title(f'distribution of {feature}')
    ax.set_ylabel('Frequency')
    plt.show()


    
def scatter_feature_distribution(tx, y, feature_1, feature_2, filter=None):

    features = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
       'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
    
    idx = np.arange(30)
    feature_by_index = {features[i]: idx[i] for i in range(30)}

    data_1 = tx.T[feature_by_index[feature_1]]
    data_2 = tx.T[feature_by_index[feature_2]]
    
    data_1_0 = data_1[np.where(y == 0)]
    data_1_1 = data_1[np.where(y == 1)]
    data_2_0 = data_2[np.where(y == 0)]
    data_2_1 = data_2[np.where(y == 1)]

    fig, ax = plt.subplots(figsize=(10,10))
    

    if filter == 1:
        ax.scatter(data_1_1, data_2_1, color='red', alpha=0.5, label='1')
    elif filter == 0:
        ax.scatter(data_1_0, data_2_0, color='blue', alpha=0.5, label='0')
    else: 
        ax.scatter(data_1_1, data_2_1, color='red', alpha=0.5, label='1')
        ax.scatter(data_1_0, data_2_0, color='blue', alpha=0.5, label='0')


    ax.legend()
    ax.set_title(f'scatter plot of {feature_1} and {feature_2}')
    ax.set_xlabel(f'{feature_1}')
    ax.set_ylabel(f'{feature_2}')
    plt.show()




