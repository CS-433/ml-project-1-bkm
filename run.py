from data_cleanup import *
from implementations import *
from helpers import *
import logging as log

# hiding numpy lstsq error messages
np.warnings.filterwarnings('ignore')

# log config
logger = log.getLogger()  
logger.setLevel(log.INFO)
logger_handler = log.StreamHandler()  
logger.addHandler(logger_handler)
logger_handler.setFormatter(log.Formatter('%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s'))


log.info('loading train dataset from csv')
DATA_TRAIN_PATH = 'data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
log.info('training dataset loaded')

log.info('preprocessing train data, removing outliers, split data by number of jets')
xs, ys = preprocess_train(tX, y)

# setting hyperparameters
initial_w = np.zeros(30)
gamma = 0.03
max_iters = 2
batch_size = 2000
ratio = 0.9

weights = []
loss = []
for i in range(4):
    log.info(f'Initial training on PRI_jet_num={i} using Logistic Regression')
    w , l = logistic_regression_newton_method(ys[i], xs[i], initial_w, max_iters, batch_size, ratio, gamma)

    log.info(f'remove 5% of the misclassified points')
    dist = np.dot(xs[i], w)
    y_dist = dist*[-1 if x==0 else 1 for x in list(ys[i])]
    val = np.percentile(y_dist, 5)
    x_new = xs[i][(~(y_dist < val))]
    y_new = ys[i][(~(y_dist < val))]

    log.info(f'number of observations: NEW MODEL: {xs[i].shape[0]}, OLD MODEL: {x_new.shape[0]} data')
    log.info(f'Training second model on PRI_jet_num={i} using Logistic Regression')
    w , l = logistic_regression_newton_method(y_new, x_new, w, max_iters, batch_size, ratio, gamma)

    log.info(f'remove 5% of the misclassified points')
    dist = np.dot(x_new, w)
    y_dist = dist*[-1 if x==0 else 1 for x in list(y_new)]
    val = np.percentile(y_dist, 5)
    x_new2 = x_new[(~(y_dist < val))]
    y_new2 = y_new[(~(y_dist < val))]

    log.info(f'number of observations: NEW MODEL: {x_new2.shape[0]}, OLD MODEL: {x_new.shape[0]} data')
    log.info(f'Training third model on PRI_jet_num={i} using Logistic Regression')
    w , l = logistic_regression_newton_method(y_new2, x_new2, w, max_iters, batch_size, ratio, gamma)
    weights.append(w)
    loss.append(l)


log.info('loading test dataset from csv')
OUTPUT_PATH = 'data/test.csv' 
_, tX_test, ids_test = load_csv_data(OUTPUT_PATH)
log.info('test dataset loaded')

log.info('preprocessing test data, removing outliers, split data by number of jets')
tx_test, idx = preprocess_test(tX_test, ids_test)

log.info('making prediction on test data')
labels = []
for i in range(4):
    labels.append(predict_logistic_labels(weights[i], tx_test[i]))
final = np.array(list(labels[0])+list(labels[1])+list(labels[2])+list(labels[3]))
final = final * 2 - 1
final_idx = np.array(list(idx[0])+list(idx[1])+list(idx[2])+list(idx[3]))

log.info('dumping results in csv')
create_csv_submission(final_idx, final, 'bkm_sample_submission.csv')
log.info('results are dumped')