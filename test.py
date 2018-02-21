import dataset_mnist
import softmaxreg

X_train, y_train, X_test, y_test = dataset_mnist.load_mnist()
#X_train, y_train, X_test, y_test = dataset_mnist.load_mini_mnist()

softmaxreg.train(X_train, y_train, X_test, y_test, 100, 0.00002, lbda = 0.1, use_intercept = True)
