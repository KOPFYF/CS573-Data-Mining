import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data
# X = np.loadtxt('toy_PCAdata.txt', delimiter=',')
X = np.loadtxt('PCAdata.txt', delimiter=',')
XX = np.matmul(X.T, X)
print(X.shape, X)
print(XX.shape, XX)
N, D = X.shape[0], X.shape[1]
M = 10
layer1_neurons = M

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
        self.fc1_U = nn.Linear(D, layer1_neurons, bias = False)

        self.act = nn.Softmax()
        # Fully connected layer 2 user
        self.fc2_U = nn.Linear(layer1_neurons, M, bias = False)


    def forward(self, x1, x2):
        
        x1 = self.fc1_U(x1)
        x1 = self.act(x1)
        x1 = self.fc2_U(x1)
        
        x2 = self.fc1_U(x2)
        x2 = self.act(x2)
        x2 = self.fc2_U(x2)
      
        y_hat = (x1*x2).sum()
        # y_hat = self.fc2_U(h)
        # uut = torch.matmul(self.fc1_U.weight.t(),self.fc1_U.weight)
        # print(f"y_hat={y_hat}")

        return y_hat

    def get_U(self):
        return self.fc1_U.weight

    def get_gamma(self):
        return self.fc2_U.weight

net = Net()
print(net)


#import torch.optim as optim

total_steps = 100000
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
            # print(y_hat.data.numpy())

            loss = loss + mse_loss(y_hat,xx)
            # loss = loss + mse_loss(y_hat,xx) + mse_loss(uut,Unity)          
            print('Loss:',loss.data.numpy())

    loss.backward()
    opt_adam.step()

U = net.get_U().data.numpy()
gamma = net.get_gamma().data.numpy()
gamma = np.diag(gamma[0])

# gamma_sqrt = np.sqrt(gamma)
# gamma_sqrt_inv = np.linalg.inv(gamma_sqrt)
# Z_nn = np.dot(np.dot(X,U.T),gamma_sqrt_inv)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=M, copy=True, whiten=False)
# Z_sk = pca.fit(Z_nn)
# F_norm  = Z_nn - Z_sk


