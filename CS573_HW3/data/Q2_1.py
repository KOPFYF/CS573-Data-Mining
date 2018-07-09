import torch
import torch.autograd as autograd
import numpy as np
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# This method creates a pytorch model


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
    inputs = []

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    X = autograd.Variable(torch.from_numpy(train_x),
                          requires_grad=True).float()
    Y = autograd.Variable(torch.from_numpy(train_y),
                          requires_grad=False).long()

    for itr in range(epoch):
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
    test_y = test_y.values[:, 0]

    print(y_pred.shape, test_y.shape)
    # print(y_pred.data.numpy())
    print(idx.data.numpy())
    print(test_y)
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


def plot_AUC(train_frac, AUC_result):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.plot(train_frac, AUC_result)
    plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
           [r'$10%$', r'$30%$', r'$50%$', r'$70%$', r'$90%$' , r'$100%$'])
    plt.xlabel('training examples fraction')
    plt.ylabel('AUC')
    plt.show()
    fig.savefig('AUC_q2_1.png')

def main():

    # You are provided a skelton pytorch nn code for your experiments
    # Use this main code to understand how the methods are used
    # There are two methods nn_train, which is used to train a model (created using create_model method)
    # Other than data nn train takes learning rate and number of iterations as parameters.
    # Please change the given values for your experiments.
    # nn_train returns a trained model
    # pass this model to nn_test to get accuracy
    # nn_test return 0/1 accuracy
    # Please read pytorch documentation if you need more details on implementing nn using pytorch
    # training and test data is read as numpy arrays

    # You can use python pandas if you want to fasten reading file , the code
    # is commented
    df = pd.read_csv("test.csv")
    test_x = df.drop(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'], axis=1)
    test_x = test_x.values
    df = pd.read_csv("pred.csv")
    test_y = df.drop(['Loan ID'], axis=1)

    test_x[np.isnan(test_x)] = 0
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # train_x = df.drop(['Loan ID','Status (Fully Paid=1, Not Paid=0)'], axis = 1)
    # train_y = df['Status (Fully Paid=1, Not Paid=0)']
    # train_x = train_x.values
    # train_y = train_y.values

    idim = 25  # input dimension
    hdim1 = 64  # 64 # hidden layer one dimension
    hdim2 = 64  # 64 # hidden layer two dimension
    odim = 2   # output dimension

    AUC_result = []
    df_train = pd.read_csv("train.csv")
    train_frac = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i in train_frac:
        df = df_train.sample(frac=i, replace=False)
        train_x = df.drop(
            ['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'], axis=1)
        train_y = df['Status (Fully Paid=1, Not Paid=0)']
        train_x = train_x.values
        train_y = train_y.values
        train_x[np.isnan(train_x)] = 0
        print(train_x.shape, train_y.shape)
        # train_x = normalize(train_x)

        # creating model structure
        model = create_model(idim, odim, hdim1, hdim2)
        trained_model = nn_train(
            train_x, train_y, model, 100, 0.01)  # training model
        AUC0, AUC = nn_test(test_x, test_y, trained_model)  # testing model
        # print('AUC0', AUC0, 'AUC', AUC)
        AUC_result.append(AUC)
    print(AUC_result)
    plot_AUC(train_frac, AUC_result)
    thefile = open('AUC_result.txt', 'w')
    for item in AUC_result:
        thefile.write("%s\n" % item)

main()
