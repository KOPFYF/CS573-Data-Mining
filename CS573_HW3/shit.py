from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pylab
import numpy as np

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)
# print(X_train.shape)
# print(y_train.shape)
# y_train.reshape(1,-1)
# print(y_train.shape)
# print(type(y_train))
print(y_test)

# pylab.scatter(X[:,0], X[:,1], c=y)
# pylab.show()

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[1] # size of input layer
    n_h1 = 2
    n_h2 = 2
    n_y = 1 # size of output layer
    return (n_x, n_h1, n_h2, n_y)


def init_weights(n_x = 2, n_h1 = 2, n_h2 = 2, n_y = 1):
    """
    Argument:
    n_x -- size of the input layer
    n_h1 -- size of the hidden layer 1
    n_h2 -- size of the hidden layer 2
    n_y -- size of the output layer
    
    Returns:
    model -- python dictionary containing your model:
                    W1 -- weight matrix of shape (n_h1, n_x)
                    b1 -- bias vector of shape (n_h1, 1)
                    W2 -- weight matrix of shape (n_h2, n_h1)
                    b2 -- bias vector of shape (n_h2, 1)
                    W3 -- weight matrix of shape (n_y, n_h2)
                    b3 -- bias vector of shape (n_y, 1)
    """
    model = dict(
        W1 = np.random.randn(n_h1, n_x),
        b1 = np.zeros(shape=(n_h1, 1)),
        W2 = np.random.randn(n_h2, n_h1),
        b2 = np.zeros(shape=(n_h2, 1)),
        W3 = np.random.randn(n_y, n_h2),
        b3 = np.zeros(shape=(n_y, 1))
    )

    return model

# (n_x, n_h1, n_h2, n_y) = layer_sizes(X_train, y_train)
# model = init_weights(n_x, n_h1, n_h2, n_y)


# Defines the softmax function. For two classes, this is equivalent to the logistic regression
def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def ReLu(Z):
    A = Z
    A[Z < 0] = 0
    return A

# For a single example $x$
def forward(x, model):
    
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    W3 = model['W3']
    b3 = model['b3']

    # Input times first layer matrix 
    Z1 = np.dot(W1, x.T) + b1

    # ReLU activation goes to hidden layer
    A1 = ReLu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = ReLu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = np.tanh(Z3)
    # A3 = softmax(Z3.T).T

    # Hidden layer values to output
    # hat_y = softmax(Z3.T).T

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}

    return cache, A3

# cache, hat_y = forward(X_train, model)
# print(cache['A3'].shape, cache['A2'].shape, cache['A1'].shape)
def compute_cost(A3, Y, model):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[0] # number of example
    
    # Retrieve W1 and W2 from parameters
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = model['W1']
    W2 = model['W2']
    W3 = model['W3']
    ### END CODE HERE ###
    
    # Compute the cross-entropy cost
    ### START CODE HERE ### (≈ 2 lines of code)
    logprobs = np.multiply(np.log(A3), Y.T) + np.multiply((1 - Y.T), np.log(1 - A3))
    cost = - np.sum(logprobs) / m
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
    
    return cost


def backward(model, cache, X, Y):
    """xs, hs, errs contain all information (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    m = X.shape[0]

    W1 = model['W1']
    W2 = model['W2']
    W3 = model['W3']

    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']

    dZ3 = A3 - Y.T
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dZ2 = dA2
    dZ2[A2 < 0] = 0

    dW2 = (1/m) * np.dot(dZ2, A1.T)/m   
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = dA1
    dZ1[A1 < 0] = 0

    dW1 = (1/m) * np.dot(dZ1, X)/m    
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}
    
    return grads

# grads = backward(model, cache, X_train, y_train)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

def gradient_step(model, grads, learning_rate = 1e-1):

    # Update every model in our networks (W1 and W2) using their gradients
    for para1, para2 in zip(grads, model):
        # Careful, learning rate should depend on mini-batch size
        model[para2] += learning_rate * grads[para1]

    return model

def gradient_descent(X_train, y_train, no_iter=10, n_h1 = 10, n_h2 = 10):
    minibatch_size = 50
    model = init_weights(2, n_h1, n_h2, 1)
    W1 = model['W1']
    W2 = model['W2']
    W3 = model['W3']
    b1 = model['b1']
    b2 = model['b2']
    b3 = model['b3']

    for i in range(0, no_iter):
        print('Iteration (epoch) {}'.format(iter))
        cache, A3 = forward(X_train, model)
        
        cost = compute_cost(A3, y_train, model)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward(model, cache, X_train, y_train)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        model = gradient_step(model, grads, learning_rate = 1e-1)
        
        # # Print the cost every 1000 iterations
        # if print_cost and i % 1000 == 0:
        #     print ("Cost after iteration %i: %f" % (i, cost))

    return model

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    cache, A3 = forward(X, parameters)
    predictions = np.round(A3)
    
    return predictions

model = gradient_descent(X_train, y_train, no_iter=10, n_h1=100, n_h2=100)
predictions = predict(model, X_test)
print(predictions)
print(predictions.shape)
print ('Accuracy: %d' % float((np.dot(y_test.T, predictions.T) + np.dot(1 - y_test.T, 1 - predictions.T)) / float(y_test.size) * 100) + '%')
# model = gradient_step(model, cache, X_train, y_train, learning_rate = 1e-1)

# print("W1 = " + str(model["W1"]))
# print("b1 = " + str(model["b1"]))
# print("W2 = " + str(model["W2"]))
# print("b2 = " + str(model["b2"]))

# def gradient_descent(model, X_train, y_train, no_iter=10):

#     minibatch_size = 50

#     (n_x, n_h1, n_h2, n_y) = layer_sizes(X_train, y_train)
#     model = init_weights(n_x, n_h1, n_h2, n_y)
    
#     for iter in range(no_iter):
#         print('Iteration (epoch) {}'.format(iter))

#         ## MINI-BATCH: Shuffles the training data to sample without replacement
#         indices = list(range(0,X_train.shape[0]))
#         np.random.shuffle(indices)
#         X_train = X_train[indices,:]
#         y_train = y_train[indices]

#         for i in range(0, X_train.shape[0], minibatch_size):
#             # Get pair of (X, y) of the current mini-batch
#             X_train_mini = X_train[i:i + minibatch_size]
#             y_train_mini = y_train[i:i + minibatch_size]

#             cache, hat_y = forward(X_train_mini, model)
#             # grads = backward(model, cache, X_train_mini, y_train_mini)
#             model = gradient_step(model, cache, X_train_mini, y_train_mini, learning_rate = 1e-1)

#     return model

# no_iter = 10

# # Reset model
# model = init_weights()

# # Train the model
# model = gradient_descent(model, X_train, y_train, no_iter=no_iter)

# y_pred = np.zeros_like(y_test)

# accuracy = 0

# for i, x in enumerate(X_test):
#     # Predict the distribution of label
#     _, prob = forward(x, model)
#     # print(prob)
#     # Get label by picking the most probable one
#     y = np.argmax(prob)
#     y_pred[i] = y

#     # Accuracy of predictions with the true labels and take the percentage
#     # Because our dataset is balanced, measuring just the accuracy is OK
#     accuracy = (y_pred == y_test).sum() / y_test.size

# print('Accuracy after {} iterations: {}'.format(no_iter,accuracy))
# pylab.scatter(X_test[:,0], X_test[:,1], c=y_pred)
# pylab.show()

# no_iter = 10
# no_runs = 10

# accuracies = np.zeros(no_runs)

# for run in range(no_runs):
#     print("Run {}".format(run))
#     # Reset model
#     model = init_weights()

#     # Train the model
#     model = gradient_descent(model, X_train, y_train, no_iter=no_iter)

#     y_pred = np.zeros_like(y_test)
    
#     for i, x in enumerate(X_test):
#         # Predict the distribution of label
#         _, prob = forward(x, model)
#         # Get label by picking the most probable one
#         y = np.argmax(prob)
#         y_pred[i] = y

#         # Accuracy of predictions with the true labels and take the percentage
#         # Because our dataset is balanced, measuring just the accuracy is OK
#         accuracies[run]= (y_pred == y_test).sum() / y_test.size

# print('Mean accuracy over test data: {}, std: {}'.format(accuracies.mean(), accuracies.std()))


