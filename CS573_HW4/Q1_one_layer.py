import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
X = np.loadtxt('toy_PCAdata.txt', delimiter=',')
# X = np.loadtxt('PCAdata.txt', delimiter=',')
XX = np.matmul(X.T, X)
print(X.shape, X)
print(XX.shape, XX)
N, D = X.shape[0], X.shape[1]

layer1_neurons = D

M = D

def get_one_hot(idx,dim):
    onehot = torch.zeros(dim).float()
    onehot[idx] = 1.0
    return onehot

from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch
from torch.optim import optimizer


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Fully connected layer 1 user
        self.fc1_U = nn.Linear(D, M, bias = False)
        # Fully connected layer 2 user
        # self.fc2_U = nn.Linear(layer1_neurons, M, bias = False)

        # self.unit = nn.softmax()


    def forward(self, x1, x2):
        
        x1 = self.fc1_U(x1)
        # self.fc1_U.weight = torch.norm(self.fc1_U.weight)
        # x1 = self.unit(x1)
        # x1 = self.fc2_U(x1)
        
        x2 = self.fc1_U(x2)
        # x2 = self.unit(x2)
        # x2 = self.fc2_U(x2)
      
        y_hat = (x1*x2).sum()
        # y_hat = self.fc2_U(x)
        # uut = torch.matmul(self.fc1_U.weight.t(),self.fc1_U.weight)
        # print(f"y_hat={y_hat}")

        return y_hat

    def get_U(self):
        return self.fc1_U.weight


net = Net()
print(net)


#import torch.optim as optim

total_steps = 1000
learning_rate = 1e-2

mse_loss = nn.MSELoss()


opt_adam = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.99))

for step in range(total_steps):
    print(f"Step {step}")
    
    net.zero_grad()
    # opt_SGD.zero_grad()
    loss = Variable(torch.from_numpy(np.array([0])).float())

    for i in range(D):
        for j in range(D):
            x1 = Variable(get_one_hot(int(i),dim=D), requires_grad=True).float()
            x2 = Variable(get_one_hot(int(j),dim=D), requires_grad=True).float()
            xx = Variable(torch.from_numpy(np.array([XX[i][j]])), requires_grad=False).float()
            # Unity = Variable(torch.from_numpy(np.identity(D)), requires_grad=False).float()

            y_hat = net(x1,x2)
            # y_hat,uut = net(x1,x2)
            print(y_hat.data.numpy())

            loss = loss + mse_loss(y_hat,xx)
            # loss = loss + mse_loss(y_hat,xx) + mse_loss(uut,Unity)          
            print('Loss:',loss.data.numpy())

    loss.backward()
    opt_adam.step()

U = net.get_U().data.numpy()
# gamma_sqrt = net.get_gamma_sqrt().data.numpy()
# gamma_sqrt = np.diag(gamma_sqrt[0])
# gamma = np.dot(gamma_sqrt,gamma_sqrt)
# print('gamma',gamma)
print('U',U)
print('U*UT',np.dot(U,U.T))
# print('U*gamma*UT',np.dot(np.dot(U,gamma),U.T))
print('XT*X',XX)

# gamma_sqrt = np.sqrt(gamma)
# gamma_sqrt_inv = np.linalg.inv(gamma_sqrt)
# Z_nn = np.dot(np.dot(X,U),gamma_sqrt_inv)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=10, copy=True, whiten=False)
# Z_sk = pca.fit(Z_nn)
# F_norm  = Z_nn - Z_sk

