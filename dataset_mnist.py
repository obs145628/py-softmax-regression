from sklearn.datasets import fetch_mldata
import numpy as np
import pickle
from matplotlib import pyplot as plt

TRAIN_TEST_FILE = './mnist_60k_10k.bin'
MINI_TRAIN_TEST_FILE = './mnist_1k_100.bin'
BIN_TRAIN_TEST_FILE = './mnist_bin_60k_10k.bin'

def digit_to_vec(x):
    res = np.zeros(10, dtype=np.float32)
    res[int(x)] = 1
    return res

def vec_to_digit(x):
    return np.argmax(x)

def sep_train_test(data, target, train_len, test_len):
    total_len = train_len + test_len

    arr = list(zip(data, target))
    np.random.shuffle(arr)
    arr = arr[:total_len]
    train = arr[:train_len]
    test = arr[train_len:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def build_data_train_test(path, train_len, test_len):
    print('Downloading dataset...')
    mnist = fetch_mldata('MNIST original')
    data = sep_train_test(mnist.data, mnist.target, train_len, test_len)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print('Dataset downloaded')
    return data

def load_data_train_test(path, train_len, test_len):
    try:
        f = open(path, 'rb')
    except IOError:
        return build_data_train_test(path, train_len, test_len)
    with f:
        return pickle.load(f)


def load_train_test(path, train_len, test_len):
    X_train, y_train, X_test, y_test = load_data_train_test(path, train_len, test_len)

    X_train = X_train / 255
    X_test = X_test / 255
    y_train = np.array([digit_to_vec(y) for y in y_train])
    y_test = np.array([digit_to_vec(y) for y in y_test])
    return X_train, y_train, X_test, y_test

def load_mnist():
    return load_train_test(TRAIN_TEST_FILE, 60000, 10000)

def load_mini_mnist():
    return load_train_test(MINI_TRAIN_TEST_FILE, 1000, 100)


def show_digit(x, y):
    n = vec_to_digit(y)
    pixels = (x * 255).reshape((28, 28)).astype(np.uint8)
    plt.imshow(pixels, cmap='gray')

    plt.title('mnist number ' + str(n))
    plt.show()


def output_test(y_hat, y):
    return vec_to_digit(y_hat) == vec_to_digit(y)







#### Binary case: only 0 and 1

def bin_sep_train_test(data, target, train_len, test_len):
    total_len = train_len + test_len
    arr = list(zip(data, target))    


    arr = [x for x in arr if x[1] == 0 or x[1] == 1]
    
    np.random.shuffle(arr)
    arr = arr[:total_len]
    train = arr[:train_len]
    test = arr[train_len:]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)




def bin_build_data_train_test(path, train_len, test_len):
    print('Downloading dataset...')
    mnist = fetch_mldata('MNIST original')
    data = bin_sep_train_test(mnist.data, mnist.target, train_len, test_len)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print('Dataset downloaded')
    return data

def bin_load_data_train_test(path, train_len, test_len):
    try:
        f = open(path, 'rb')
    except IOError:
        return bin_build_data_train_test(path, train_len, test_len)
    with f:
        return pickle.load(f)


def bin_load_train_test(path, train_len, test_len):
    X_train, y_train, X_test, y_test = bin_load_data_train_test(path, train_len, test_len)

    X_train = X_train / 255
    X_test = X_test / 255
    y_train = np.array([int(y) for y in y_train])
    y_test = np.array([int(y) for y in y_test])
    return X_train, y_train, X_test, y_test


def load_mnist_bin():
    #14780
    return bin_load_train_test(BIN_TRAIN_TEST_FILE, 12000, 2780)
