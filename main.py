import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.datasets import mnist
import pickle

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
SCALE_FACTOR = 255
WIDTH = X_train.shape[1]
HEIGHT = X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR

def relu(Z):
  return np.maximum(Z, 0)

def deriv_relu(Z):
  return Z > 0

def softmax(Z):
  exp = np.exp(Z - np.max(Z))
  return exp / exp.sum(axis=0)

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y

def init_params(size):
  W1 = np.random.rand(10, size) - 0.5
  b1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10, 10) - 0.5
  b2 = np.random.rand(10, 1) - 0.5
  return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = relu(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

def backward_prop(Z1, A1, A2, W2, X, Y, m):
  one_hot_Y = one_hot(Y)
  dZ2 = 2*(A2 - one_hot_Y)
  dW2 = 1 / m * (dZ2.dot(A1.T))
  db2 = 1 / m * np.sum(dZ2, 1)
  dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1)
  dW1 = 1 / m * (dZ1.dot(X.T))
  db1 = 1 / m * np.sum(dZ1, 1)
  return dW1, db1, dW2, db2,

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * np.reshape(db1, (10,1))
  W2 = W2 - alpha * dW2
  b2 = b2 - alpha * np.reshape(db2, (10,1))
  return W1, b1, W2, b2

def get_predictions(A2):
  return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
  print(predictions, Y)
  return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
  size , m = X.shape
  # W1, b1, W2, b2 = init_params(size)
  with open("trained_params.pkl","rb") as dump_file:
    W1, b1, W2, b2=pickle.load(dump_file)
  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y, m)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    if i % 50 == 0:
      print("Iteration: ", i)
      print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
  return W1, b1, W2, b2


def make_predictions(X, W1 ,b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2):
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)

with open("trained_params.pkl","wb") as dump_file:
    pickle.dump((W1, b1, W2, b2),dump_file)

with open("trained_params.pkl","rb") as dump_file:
    W1, b1, W2, b2=pickle.load(dump_file)
show_prediction(0,X_test, Y_test, W1, b1, W2, b2)
show_prediction(1,X_test, Y_test, W1, b1, W2, b2)
show_prediction(2,X_test, Y_test, W1, b1, W2, b2)
show_prediction(100,X_test, Y_test, W1, b1, W2, b2)
show_prediction(200,X_test, Y_test, W1, b1, W2, b2)
