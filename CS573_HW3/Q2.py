import torch
import torch.autograd as autograd
import numpy as np
import torch.optim as optim

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

# This method trains a model with the given data
# Epoch is the number of training iterations
# lrate is the learning rate
def nn_train(train_x, train_y, model, epoch, lrate):
    inputs = []


    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    X = autograd.Variable(torch.from_numpy(train_x), requires_grad=True).float()
    Y = autograd.Variable(torch.from_numpy(train_y), requires_grad=False).long()

    for itr in range(epoch):

        y_pred = model(X)

        loss = loss_fn(y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        print("Epoch: {}  Acc: {}".format(itr,nn_test(test_x,test_y,model)))

    return model

# Pass the trained model along with test data to this method to get accuracy
# The method return accuracy value
def nn_test(test_x, test_y, model):

    X = autograd.Variable(torch.from_numpy(test_x), requires_grad=False).float()
    y_pred = model(X)
    _ , idx = torch.max(y_pred, 1)

#    test_y = test_y.values[:,0]
    return (1.*np.count_nonzero((idx.data.numpy() == test_y).astype('uint8')))/len(test_y)


def normalize(data_list):
    # compute normalization parameter
    utter = np.concatenate(data_list, axis=0)
    mean  = np.mean(utter)
    utter -= mean
    std   = np.std(utter)
    utter /= std

    # normalize data
    for data in data_list:
        data -= mean
        data /= std

    return data_list

# Method reads training data and drops loan id field
def read_train_data(filename):
    f = open(filename, "r")
    x = []
    y = []
    content = f.readlines() 

    for i in range(1, len(content)):
        line = content[i]
        line.strip()
        line = line.split(",")
        y.append(line[len(line)-1])
        line = line[1:len(line)-1]
        x.append(line)


    x = np.array(x)
    x = x.astype(np.float)

    y = np.array(y)
    y = y.astype(np.int)
    return (x,y)



# Method reads features of test data
def read_test_data_x(filename):
    f = open(filename, "r")
    x = []
    content = f.readlines() 

    for i in range(1, len(content)):
        line = content[i]
        line.strip()
        line = line.split(",")
        line = line[1:len(line)-1]
        x.append(line)


    x = np.array(x)
    x = x.astype(np.float)

    return x

# Method reads labels of test data
def read_test_data_y(filename):
    f = open(filename, "r")
    y = []
    content = f.readlines() 

    for i in range(1, len(content)):
        line = content[i]
        line.strip()
        line = line.split(",")
        y.append(line[len(line)-1])



    y = np.array(y)
    y = y.astype(np.int)
    return y




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
    train_x, train_y = read_train_data("train.csv")
    test_x = read_test_data_x("test.csv")
    test_y = read_test_data_y("pred.csv")


  # You can use python pandas if you want to fasten reading file , the code is commented
  #  df = pd.read_csv("train.csv")
  #  train_x = df.drop(columns = ['Loan ID','Status (Fully Paid=1, Not Paid=0)'], axis = 1)
  #  train_y = df['Status (Fully Paid=1, Not Paid=0)']
  #  train_x = train_x.values
  #  train_y = train_y.values

  #  df = pd.read_csv("test.csv")
  #  test_x = df.drop(columns = ['Loan ID','Status (Fully Paid=1, Not Paid=0)'], axis = 1)
  #  test_x = test_x.values
  #  df = pd.read_csv("pred.csv")
  #  test_y = df.drop(columns = ['Loan ID'], axis = 1)

    train_x[np.isnan(train_x)] = 0
    test_x[np.isnan(test_x)] = 0
    #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    train_x = normalize(train_x)
    idim = 25  # input dimension
    hdim1 = 64 # hidden layer one dimension
    hdim2 = 64 # hidden layer two dimension
    odim = 2   # output dimension

    model = create_model(idim, odim, hdim1, hdim2) # creating model structure
    trained_model = nn_train(train_x, train_y, model, 100, 0.01) # training model
    print(nn_test(test_x, test_y, trained_model)) # testing model

main()