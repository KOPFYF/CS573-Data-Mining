import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load movie recommendation data
dataset = pd.read_csv("u.data",sep='\t',names="user_id,item_id,rating,timestamp".split(","))
print(dataset.head())

print((len(dataset.user_id.unique()), len(dataset.item_id.unique())))

from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.99)

layer1_neurons = 20
layer2_neurons = 2
n_users, n_movies = len(dataset.user_id.unique()), len(dataset.item_id.unique())

# creates a one-hot encoding of the indices
def get_one_hot(idx,dim):
    onehot = torch.zeros(dim).float()
    onehot[idx] = 1.0
    return onehot

from torch.autograd import Variable
from torch import nn
import torch


class Net_Wrong(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Fully connected layer 1 user
        self.fc1_user = nn.Linear(n_users, layer1_neurons)
        # Fully connected layer 2 user
        self.fc2_user = nn.Linear(layer1_neurons, layer2_neurons)
        # Fully connected layer 1 movie
        self.fc1_movie = nn.Linear(n_movies, layer1_neurons)
        # Fully connected layer 2 movie
        self.fc2_movie = nn.Linear(layer1_neurons, layer2_neurons)

    def forward(self, x_user, x_movie):
        
        x_user = self.fc1_user(x_user)
        x_user = self.fc2_user(x_user)
        #print(f"x_user={x_user}")

        x_movie= self.fc1_movie(x_movie)
        x_movie= self.fc2_movie(x_movie)
        #print(f"x_movie={x_movie}")
        
        y_hat = (x_user*x_movie).sum()
        #print(f"y_hat={y_hat}")
        return y_hat

from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Fully connected layer 1 user
        self.fc1_user = nn.Linear(n_users, layer1_neurons)
        # Fully connected layer 2 user
        self.fc2_user = nn.Linear(layer1_neurons, layer2_neurons)
        # Fully connected layer 1 movie
        self.fc1_movie = nn.Linear(n_movies, layer1_neurons)
        # Fully connected layer 2 movie
        self.fc2_movie = nn.Linear(layer1_neurons, layer2_neurons)

    def forward(self, x_user, x_movie):
        
        x_user = F.tanh(self.fc1_user(x_user))
        x_user = self.fc2_user(x_user)
        #print(f"x_user={x_user}")

        x_movie= F.tanh(self.fc1_movie(x_movie))
        x_movie= self.fc2_movie(x_movie)
        #print(f"x_movie={x_movie}")
        
        y_hat = (x_user*x_movie).sum()
        #print(f"y_hat={y_hat}")
        return y_hat
    
    def emb_user(self, x_user):
        x_user = F.relu(self.fc1_user(x_user))
        x_user = self.fc2_user(x_user)
        return x_user

net = Net()
print(net)

#import torch.optim as optim

total_steps = 20
learning_rate = 1e-2

mse_loss = nn.MSELoss()

for step in range(total_steps):
    print(f" Step {step}")
    
    net.zero_grad()
    #optimizer.zero_grad()

    for user, movie, rating in zip(train.user_id,train.item_id,train.rating):
        # prepare input
        x_user = Variable(get_one_hot(int(user-1),dim=n_users)).float()
        x_movie = Variable(get_one_hot(int(movie-1),dim=n_movies)).float()
        y_rating = Variable(torch.from_numpy(np.array([rating])).float())

        # forward pass
        y_hat = net(x_user,x_movie)

        # compute Mean Squared Error loss
        loss = mse_loss(y_hat,y_rating)
        
        #print(f"Values: {y_rating.data[0], y_hat.data[0], loss.data[0]}")
        
        # Add the gradients of this example to the list of gradients computed via backpropagation
        loss.backward()
    
    #optimizer.step() 
    # Gradient all computed, time to update gradients
    for f in net.parameters(): 
        #print(f.grad.data)
        f.data.sub_(f.grad.data/len(train.user_id) * learning_rate) # subtract because we are minimizing the Mean Squared Error (MSE)

def user_embedding(user_id):
    x_user = Variable(get_one_hot(int(user_id-1),dim=n_users)).float()
    u_emb = net.emb_user(x_user)
    return u_emb.data.numpy()

import pylab

emb_matrix = np.zeros((n_users,2))
for user_id in train.user_id:
    emb_matrix[int(user_id-1),:] = user_embedding(user_id)


pylab.scatter(emb_matrix[np.all(emb_matrix!=0, axis=1),0], emb_matrix[np.all(emb_matrix!=0, axis=1),1])

#plt.xlim(0.6, 0.9)
#plt.ylim(-0.1, -0.5)

plt.show()
