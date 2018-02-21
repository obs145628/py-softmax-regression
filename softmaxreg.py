'''
Softmax regression implementation

More informations:
- http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
- https://medium.com/@awjuliani/simple-softmax-in-python-tutorial-d6b4c4ed5c16
'''


import numpy as np
import dataset_mnist


'''
Compute matrix version of softmax
compute the sofmax functon for each row
@param x matrix of size (set_len, output_len)
@return matrix of size (set_len, output_len)

softmax(x)_i = exp(x_i) / (sum_(j=1->output_len) exp(x_j))
'''
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x.T / e_x.sum(axis=1)).T # only difference


'''
Compute cost function
@param x matrix of size (set_len, input_len)
@param y matrix of size (set_len, output_len)
@param w matrix of size (input_len, output_len)
@param lbda - coefficient for l2 regularization
@return cost value

'''
def cost_function(X, y, w, lbda):
    z = np.dot(X, w)
    y_hat = softmax(z)

    res = - np.sum(y * np.log(y_hat))
    reg2 = np.sum(w * w)
    #reg2 = np.dot(x.flatten(), x.flatten()) slower method

    return res + reg2


def evaluate(X, y, w, lbda):

    y_hat = softmax(np.dot(X, w))
    total = X.shape[0]
    succ = 0
    for i in range(total):
        succ += dataset_mnist.output_test(y[i], y_hat[i])
    
    perc = succ * 100 / total

    print('Cost: ' + str(cost_function(X, y, w, lbda)))
    print('Results: {} / {} ({}%)'.format(succ, total, perc))


'''
Apply stochastic gradient descent on the whole training set of size set_len
@param x matrix of size (set_len, input_len)
@param y matrix of size (set_len, output_len)
@param w matrix of size (input_len, output_len)
@param lr - learning rate
@param lbda - coefficient for l2 regularization
@return matrix (input_len, output_len) updated weights
'''
def sgd(X, y, w, lr, lbda):
    y_hat = softmax(np.dot(X, w))
    dW = - np.dot(X.T, y - y_hat) + lbda * w
    w = w - lr * dW
    return w


'''
@param X_train - matrix (set_len, input_len) matrix of features (training set)
@param y_train - matrix (set_len, output_len) matrix of labels (training set)
@param x_test - matrix (set_len, input_len) matrix of features (test set)
@param y_test - matrix (set_len, output_len) matrix of labels (test set)
@param epochs - number of epochs of learning
@param lr - learning rate
@param lbda - coefficient for l2 regularization
@param use_intercept - if true, add a bias to the weights

Run the training for several epochs.
After each epochs, the wieghts are tested on the training test and the test set
'''
def train(X_train, y_train, X_test, y_test, epochs, lr, lbda = 0, use_intercept = False):

    if use_intercept:
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    w = np.zeros((X_train.shape[1], y_train.shape[1]))

    #Training
    for i in range(1, epochs + 1):
        print('Epoch :' + str(i))
        w = sgd(X_train, y_train, w, lr, lbda)
        print('Train:')
        evaluate(X_train, y_train, w, lbda)
        print('Test:')
        evaluate(X_test, y_test, w, lbda)

    return w
