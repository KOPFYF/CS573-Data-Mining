from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
# import pylab
import numpy as np
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# There are only two features in the data X[:,0] and X[:,1]
n_feature = 2
# There are only two classes: 0 (purple) and 1 (yellow)
n_class = 2

def init_weights(n_hidden=100):
    # Initialize weights with Standard Normal random variables
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )

    return model

# Defines the softmax function. For two classes, this is equivalent to the logistic regression
def softmax(x):
    return np.exp(x) / np.exp(x).sum()

# For a single example $x$
def forward(x, model):
    # Input times first layer matrix 
    z_1 = x @ model['W1']

    # ReLU activation goes to hidden layer
    h = z_1
    h[z_1 < 0] = 0

    # Hidden layer values to output
    hat_y = softmax(h @ model['W2'])

    return h, hat_y

def backward(model, xs, hs, errs):
    """xs, hs, errs contain all information (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    dW2 = (hs.T @ errs)/xs.shape[0]

    # Get gradient of hidden layer
    dh = errs @ model['W2'].T
    dh[hs <= 0] = 0

    dW1 = (xs.T @ dh)/xs.shape[0]

    return dict(W1=dW1, W2=dW2)

def get_gradient(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        # Create one-hot coding of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1.

        # Compute the gradient of output layer
        err = y_true - y_pred

        # Accumulate the informations of the examples
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs.append(h)
        errs.append(err)

    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs), np.array(errs))

def gradient_step(model, X_train, y_train):
    grad = get_gradient(model, X_train, y_train)
    model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        # Learning rate: 1e-1
        model[layer] += 1e-1 * grad[layer]

    return model

def gradient_descent(model, X_train, y_train, no_iter=10):
    for iter in range(no_iter):
        print('Iteration {}'.format(iter))

        model = gradient_step(model, X_train, y_train)

    return model

no_iter = 10

# Reset model
model = init_weights()

# Train the model
model = gradient_descent(model, X_train, y_train, no_iter=no_iter)

y_pred = np.zeros_like(y_test)

accuracy = 0

for i, x in enumerate(X_test):
    # Predict the distribution of label
    _, prob = forward(x, model)
    # Get label by picking the most probable one
    y = np.argmax(prob)
    y_pred[i] = y

    # Accuracy of predictions with the true labels and take the percentage
    # Because our dataset is balanced, measuring just the accuracy is OK
    accuracy = (y_pred == y_test).sum() / y_test.size

print('Accuracy after {} iterations: {}'.format(no_iter,accuracy))

plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)
plt.show()

no_iter = 10
no_runs = 10

accuracies = np.zeros(no_runs)


for run in range(no_runs):
    print("Run {}".format(run))
    # Reset model
    model = init_weights()

    # Train the model
    model = gradient_descent(model, X_train, y_train, no_iter=no_iter)

    y_pred = np.zeros_like(y_test)
    
    for i, x in enumerate(X_test):
        # Predict the distribution of label
        _, prob = forward(x, model)
        # Get label by picking the most probable one
        y = np.argmax(prob)
        y_pred[i] = y

        # Accuracy of predictions with the true labels and take the percentage
        # Because our dataset is balanced, measuring just the accuracy is OK
        accuracies[run]= (y_pred == y_test).sum() / y_test.size

print('Mean accuracy over test data: {}, std: {}'.format(accuracies.mean(), accuracies.std()))

