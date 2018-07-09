import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.autograd as autograd
import torch.optim as optim

def create_model(idim, odim, hdim1, hdim2):
    model = torch.nn.Sequential(
        torch.nn.Linear(idim, hdim1),
        torch.nn.BatchNorm1d(hdim1),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(hdim1, hdim2),
        torch.nn.BatchNorm1d(hdim1),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Linear(hdim2, odim),
        torch.nn.LogSoftmax()
    )
    return model


def nn_train(train_x, train_y, model, batch_size, lrate):
    inputs = []
    minibatch_size = batch_size

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    
    for itr in range(100):
        # print('Iteration (epoch) {}'.format(itr))
        ## MINI-BATCH: Shuffles the training data to sample without replacement
        indices = list(range(0,train_x.shape[0]))
        np.random.shuffle(indices)
        X_train = train_x[indices,:]
        y_train = train_y[indices]

        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current mini-batch
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]

            X = autograd.Variable(torch.from_numpy(X_train_mini),
                          requires_grad=True).float()
            Y = autograd.Variable(torch.from_numpy(y_train_mini),
                          requires_grad=False).long()

            y_pred = model(X)
            loss = loss_fn(y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("Epoch: {}  Acc: {}".format(itr,nn_test(test_x,test_y,model)))
    return model

# Pass the trained model along with test data to this method to get accuracy
# The method return accuracy value


def nn_test(test_x, test_y, model):

    X = autograd.Variable(torch.from_numpy(
        test_x), requires_grad=False).float()
    y_pred = model(X)
    _, idx = torch.max(y_pred, 1)
    # test_y = test_y.values[:, 0]

    # print(y_pred.shape, test_y.shape)
    # # print(y_pred.data.numpy())
    # print(idx.data.numpy())
    # print(test_y)
    AUC = roc_auc_score(test_y, idx.data.numpy())

    return (1. * np.count_nonzero((idx.data.numpy() == test_y).astype('uint8'))) / len(test_y), AUC


def normalize(data_list0):
    # compute normalization parameter
    data_list = np.copy(data_list0)
    utter = np.concatenate(data_list, axis=0)
    mean = np.mean(utter)
    utter -= mean
    std = np.std(utter)
    utter /= std

    # normalize data
    for data in data_list:
        data -= mean
        data /= std

    return data_list, mean, std

# df = pd.read_csv("test.csv")
# test_x = df.drop(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'], axis=1)
# test_x = test_x.values
# df = pd.read_csv("pred.csv")
# test_y = df.drop(['Loan ID'], axis=1)
# test_x[np.isnan(test_x)] = 0

df = pd.read_csv("train.csv")
train_x = df.drop(
    ['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'], axis=1)
train_y = df['Status (Fully Paid=1, Not Paid=0)']
train_x = train_x.values
train_y = train_y.values
train_x[np.isnan(train_x)] = 0
# print(train_x.shape, train_y.shape)
# train_x,train_x_mean,train_x_std = normalize(train_x)

idim = 25  # input dimension
hdim1 = 64  # 64 # hidden layer one dimension
hdim2 = 64  # 64 # hidden layer two dimension
odim = 2   # output dimension

kfold = 10
hyper_combo = 4
AUC_result = np.empty((kfold,hyper_combo))
skf = StratifiedKFold(n_splits=kfold, random_state=42)

for i, (train_index, test_index) in enumerate(skf.split(train_x,train_y)):
    
    train_X, test_X = train_x[train_index], train_x[test_index]
    train_Y, test_Y = train_y[train_index], train_y[test_index]

    train_X_80, test_X_20, train_Y_80, test_Y_20 = train_test_split(train_X, train_Y, random_state=58,test_size=0.2)

    model_1 = create_model(idim, odim, hdim1, hdim2)
    trained_model_1 = nn_train(
            train_X_80, train_Y_80, model_1, 100, 0.01)  # training model
    AUC0, AUC_1 = nn_test(test_X_20, test_Y_20, trained_model_1)  # testing model
    AUC_result[i,0] = AUC_1 #{'100, 0.01': AUC_1}

    model_2 = create_model(idim, odim, hdim1, hdim2)
    trained_model_2 = nn_train(
            train_X_80, train_Y_80, model_2, 100, 0.001)  # training model
    AUC0, AUC_2 = nn_test(test_X_20, test_Y_20, trained_model_2)  # testing model
    AUC_result[i,1] = AUC_2 #{'100, 0.001': AUC_2}

    model_3 = create_model(idim, odim, hdim1, hdim2)
    trained_model_3 = nn_train(
            train_X_80, train_Y_80, model_3, 1000, 0.01)  # training model
    AUC0, AUC_3 = nn_test(test_X_20, test_Y_20, trained_model_3)  # testing model
    AUC_result[i,2] = AUC_3 #{'1000, 0.01': AUC_1}

    model_4 = create_model(idim, odim, hdim1, hdim2)
    trained_model_4 = nn_train(
            train_X_80, train_Y_80, model_4, 1000, 0.001)  # training model
    AUC0, AUC_4 = nn_test(test_X_20, test_Y_20, trained_model_4)  # testing model
    AUC_result[i,3] = AUC_4 #{'1000, 0.01': AUC_1}
    print('[Fold %d/%d Prediciton, AUC score for (1000, 0.001): %s]' % (i + 1, kfold, AUC_4))

print('AUC_Matrix:')
print(AUC_result)
print('Argmax(0~3) Array:')
Arg = AUC_result.argmax(axis = 1)
print(Arg)

AUC_final_result = []
for i, (train_index, test_index) in enumerate(skf.split(train_x,train_y)):
    
    train_X, test_X = train_x[train_index], train_x[test_index]
    train_Y, test_Y = train_y[train_index], train_y[test_index]

    if Arg[i] == 0:
        model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(
            train_X, train_Y, model, 100, 0.01)  # training model
        AUC0, AUC = nn_test(test_X, test_Y, trained_model)  # testing model
        AUC_final_result.append(AUC)
        print('This fold use hyperparameter: 100, 0.01')

    elif Arg[i] == 1:
        model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(
            train_X, train_Y, model, 100, 0.001)  # training model
        AUC0, AUC = nn_test(test_X, test_Y, trained_model)  # testing model
        AUC_final_result.append(AUC)
        print('This fold use hyperparameter: 100, 0.001')

    elif Arg[i] == 2:
        model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(
            train_X, train_Y, model, 1000, 0.01)  # training model
        AUC0, AUC = nn_test(test_X, test_Y, trained_model)  # testing model
        AUC_final_result.append(AUC)
        print('This fold use hyperparameter: 1000, 0.01')

    else:
        model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(
            train_X, train_Y, model, 1000, 0.001)  # training model
        AUC0, AUC = nn_test(test_X, test_Y, trained_model)  # testing model
        AUC_final_result.append(AUC)
        print('This fold use hyperparameter: 1000, 0.001')

    print('[Fold %d/%d Prediciton, AUC score: %s]' % (i + 1, kfold, AUC))

print('Mean and std AUC score for NN:',np.asarray(AUC_final_result).mean(),np.asarray(AUC_final_result).std())

# thefile = open('Q2_4_AUC_final_10.txt', 'w')
# for item in AUC_final_result:
#     thefile.write("%s\n" % item)
