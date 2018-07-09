import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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


def nn_train(train_x, train_y, model, epoch, lrate):
    # inputs = []

    # loss_fn = torch.nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    # X = autograd.Variable(torch.from_numpy(train_x),
    #                       requires_grad=True).float()
    # Y = autograd.Variable(torch.from_numpy(train_y),
    #                       requires_grad=False).long()

    # for itr in range(epoch):
    #     y_pred = model(X)
    #     loss = loss_fn(y_pred, Y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     # print("Epoch: {}  Acc: {}".format(itr,nn_test(test_x,test_y,model)))
    # return model

    inputs = []
    minibatch_size = 200

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    
    for iter in range(epoch):
        # print('Iteration (epoch) {}'.format(iter))
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


def normalize(data_list):
    # compute normalization parameter
    utter = np.concatenate(data_list, axis=0)
    mean = np.mean(utter)
    utter -= mean
    std = np.std(utter)
    utter /= std

    # normalize data
    for data in data_list:
        data -= mean
        data /= std

    return data_list

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


idim = 25  # input dimension
hdim1 = 64  # 64 # hidden layer one dimension
hdim2 = 64  # 64 # hidden layer two dimension
odim = 2   # output dimension


AUC_result = []

kfold = 10
skf = StratifiedKFold(n_splits=kfold, random_state=42)

for i, (train_index, test_index) in enumerate(skf.split(train_x,train_y)):
    
    train_X, test_X = train_x[train_index], train_x[test_index]
    train_Y, test_Y = train_y[train_index], train_y[test_index]
    # train_X = normalize(train_X)

    model = create_model(idim, odim, hdim1, hdim2)
    trained_model = nn_train(
            train_X, train_Y, model, 100, 0.01)  # training model
    AUC0, AUC = nn_test(test_X, test_Y, trained_model)  # testing model
    AUC_result.append(AUC)

    print('[Fold %d/%d Prediciton, AUC score: %s]' % (i + 1, kfold, AUC))
print('Mean AUC score for NN:',np.asarray(AUC_result).mean(),np.asarray(AUC_result).std())

thefile = open('k_fold_AUC_result_NN.txt', 'w')
for item in AUC_result:
    thefile.write("%s\n" % item)