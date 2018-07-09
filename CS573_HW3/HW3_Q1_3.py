from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
# from sklearn.metrics import zero_one_loss
import pylab
import numpy as np

# X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.3)

# pylab.scatter(X[:,0], X[:,1], c=y)
# pylab.show()

# There are only two features in the data X[:,0] and X[:,1]
n_feature = 2
# There are only two classes: 0 (purple) and 1 (yellow)
n_class = 2


def init_weights(n_hidden1=100,n_hidden2=100):
    # Initialize weights with Standard Normal random variables
    model = dict(
        W1=np.random.randn(n_feature, n_hidden1),
        b1 = np.zeros(shape=(1, n_hidden1)),
        W2=np.random.randn(n_hidden1, n_hidden2),
        b2 = np.zeros(shape=(1, n_hidden2)),
        W3=np.random.randn(n_hidden2, n_class),
        b3 = np.zeros(shape=(1, n_class))
    )

    return model

# Defines the softmax function. For two classes, this is equivalent to the logistic regression
def softmax(x):
	# x_max = np.max(x)*np.ones_like(x)
    return np.exp(x-np.max(x)) / np.exp(x-np.max(x)).sum()

# For a single example $x$
def forward(x, model):
    # Input times first layer matrix 
    # print('x',x.shape)
    z_1 = x @ model['W1'] + model['b1']
    
    # z_1 = np.append(z_1,model['b1'])
    # print('Z1',z_1.shape)
    # ReLU activation goes to hidden layer
    h1 = z_1
    h1[z_1 < 0] = 0

    z_2 = h1 @ model['W2'] + model['b2']
    
    # z_2 = np.append(z_2,model['b2'])
    # print('Z2',z_2.shape)

    h2 = z_2
    h2[z_2 < 0] = 0

    # Hidden layer values to output
    z_3 = h2 @ model['W3'] + model['b3']
    
    # z_3 = np.append(z_3,model['b3'])
    # print('Z3',z_3.shape)  
    hat_y = softmax(z_3)

    return h1, h2, hat_y

def backward(model, xs, hs1, hs2, errs):
    """xs, hs, errs contain all information (input, hidden state, error) of all data in the minibatch"""
    # errs is the gradients of output layer for the minibatch
    # print('xs',xs.shape)
    # print('hs1',hs1.shape)
    # print('hs2',hs2.shape)
    dW3 = (hs2.T @ errs)/xs.shape[0]

    # Get gradient of hidden layer
    dh2 = errs @ model['W3'].T 
    dZ2 = dh2
    dZ2[hs2 <= 0] = 0
    # print('dZ2.shape')
    # print(dZ2.shape)
    db3 = np.sum(errs, axis=0, keepdims=True)/xs.shape[0]

    dW2 = (hs1.T @ dh2)/xs.shape[0]

    dh1 = dZ2 @ model['W2'].T
    dZ1 = dh1
    dZ1[hs1 <= 0] = 0
    db2 = np.sum(dZ2, axis=0, keepdims=True)/xs.shape[0]
    # print('dZ2.shape')
    # print(dZ2.shape)
    # print('db2.shape')
    # print(db2.shape)

    dW1 = (xs.T @ dh1)/xs.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True)/xs.shape[0]

    return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)

def get_gradient(model, X_train, y_train):
    xs, hs1, hs2, errs = [], [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h1, h2, y_pred = forward(x, model)

        # Create one-hot coding of true label
        y_true = np.zeros(n_class)
        y_true[int(cls_idx)] = 1

        # Compute the gradient of output layer
        err = y_true - y_pred

        # Accumulate the informations of the examples
        # x: input
        # h: hidden state
        # err: gradient of output layer
        xs.append(x)
        hs1.append(h1)       
        hs2.append(h2)        
        errs.append(err)
    hs1 = np.squeeze(np.array(hs1))
    hs2 = np.squeeze(np.array(hs2))
    errs = np.squeeze(np.array(errs))
    # Backprop using the informations we get from the current minibatch
    return backward(model, np.array(xs), np.array(hs1), np.array(hs2), np.array(errs))

def gradient_step(model, X_train, y_train, learning_rate = 1e-1):
    grad = get_gradient(model, X_train, y_train)
    model = model.copy()

    # Update every parameters in our networks (W1 and W2) using their gradients
    for layer in grad:
        # Careful, learning rate should depend on mini-batch size
        model[layer] += learning_rate * grad[layer]

    return model


def gradient_descent(model, X_train, y_train, no_iter=10):

    minibatch_size = 50
    
    for iter in range(no_iter):
        print('Iteration (epoch) {}'.format(iter))

        ## MINI-BATCH: Shuffles the training data to sample without replacement
        indices = list(range(0,X_train.shape[0]))
        np.random.shuffle(indices)
        X_train = X_train[indices,:]
        y_train = y_train[indices]

        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current mini-batch
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]

            model = gradient_step(model, X_train_mini, y_train_mini, learning_rate = 1e-1)

    return model

# no_iter = 10

# # Reset model
# model = init_weights(100,100)

# # Train the model
# model = gradient_descent(model, X_train, y_train, no_iter=no_iter)

# y_pred = np.zeros_like(y_test)

# accuracy = 0

# for i, x in enumerate(X_test):
#     # Predict the distribution of label
#     _, _, prob = forward(x, model)
#     # Get label by picking the most probable one
#     y = np.argmax(prob)
#     y_pred[i] = y

#     # Accuracy of predictions with the true labels and take the percentage
#     # Because our dataset is balanced, measuring just the accuracy is OK
#     accuracy = (y_pred == y_test).sum() / y_test.size

# print('Accuracy after {} iterations with new neural network: {}'.format(no_iter,accuracy))
# pylab.scatter(X_test[:,0], X_test[:,1], c=y_pred)
# pylab.show()


# Train with 3000 new data points and validate with 100 new data points at each run.
no_iter = 100
no_runs = 20

accuracies = np.zeros(no_runs)

for run in range(no_runs):
    X, y = make_moons(n_samples=3100, random_state=None, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None,test_size=1/31)

    print("Run {}".format(run))
    # Reset model
    model = init_weights()

    # Train the model
    model = gradient_descent(model, X_train, y_train, no_iter=no_iter)

    y_pred = np.zeros_like(y_test)
    
    for i, x in enumerate(X_test):
        # Predict the distribution of label
        _, _, prob = forward(x, model)
        # Get label by picking the most probable one
        y = np.argmax(prob)
        y_pred[i] = y

        # Accuracy of predictions with the true labels and take the percentage
        # Because our dataset is balanced, measuring just the accuracy is OK
        accuracies[run]= (y_pred == y_test).sum() / y_test.size

print('Mean accuracy over test data: {}, std: {}'.format(accuracies.mean(), accuracies.std()))
thefile = open('new_accuracy.txt', 'w')
for item in accuracies:
  thefile.write("%s\n" % item)