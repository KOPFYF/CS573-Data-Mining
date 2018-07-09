def batched_nn_train(X_train, y_train, model, epoch, lrate, batch_size):
    inputs = []


    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    
    for itr in range(epoch):
        print('Iteration (epoch) {}'.format(itr))
        
        ## MINI-BATCH: Shuffles the training data to sample without replacement
        indices = list(range(0,X_train.shape[0]))
        np.random.shuffle(indices)
        X_train = X_train[indices,:]
        y_train = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            # Get pair of (X, y) of the current mini-batch
            X_train_mini = X_train[i:i + batch_size]
            y_train_mini = y_train[i:i + batch_size]
            
            X = autograd.Variable(torch.from_numpy(X_train_mini), requires_grad=True).float()
            Y = autograd.Variable(torch.from_numpy(y_train_mini), requires_grad=False).long()

            y_pred = model(X)

            loss = loss_fn(y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#        print("Epoch: {}  Acc: {}".format(itr,nn_test(test_x,test_y,model)))

    return model

def ten_fold_tuning():
    train_x, train_y = read_train_data("train.csv")
    train_x[np.isnan(train_x)] = 0
    train_x_normalized,_ = normalize(train_x,[])
    kf = StratifiedKFold(n_splits=10)
    nn_aucs = np.zeros(10)
    reg_aucs = np.zeros(10)
    tree_aucs = np.zeros(10)
    rates = [0.01,0.001]
    sizes = [100,1000]
    paras = [(i,j) for i in rates for j in sizes]
    ssf = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
    for count, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        train_x_sample, test_x_sample = train_x[train_index], train_x[test_index]
        train_y_sample, test_y_sample = train_y[train_index], train_y[test_index]
        
        #reg
        trained_model = reg_train(train_x_sample, train_y_sample)
        y_pred,_ = reg_test(test_x_sample, test_y_sample, trained_model)
        reg_aucs[count] = roc_auc_score(test_y_sample,y_pred)
        
        #tree
        trained_model = tree_train(train_x_sample, train_y_sample)
        y_pred,_ = tree_test(test_x_sample, test_y_sample, trained_model)
        tree_aucs[count] = roc_auc_score(test_y_sample,y_pred)
        print(tree_aucs[count])
        
        #nn
        idim = 25  # input dimension
        hdim1 = 64 # hidden layer one dimension
        hdim2 = 64 # hidden layer two dimension
        odim = 2   # output dimension
        model = create_model(idim, odim, hdim1, hdim2) # creating model structure
        train_x_sample, test_x_sample = train_x_normalized[train_index], train_x_normalized[test_index]
        tmp_aucs = np.zeros(len(paras))
        for train_inner_index, test_inner_index in ssf.split(train_x_sample, train_y_sample):
            train_x_inner, test_x_inner = train_x_sample[train_inner_index], train_x_sample[test_inner_index]
            train_y_inner, test_y_inner = train_y_sample[train_inner_index], train_y_sample[test_inner_index]
            for i, para in enumerate(paras):
                trained_model = batched_nn_train(train_x_inner, train_y_inner, model, 100, para[0], para[1]) # training model
                y_pred,_ = nn_test(test_x_inner, test_y_inner, trained_model)
                tmp_aucs[i] = roc_auc_score(test_y_inner,y_pred)
        best_para = paras[tmp_aucs.argmax()]
        print(best_para, tmp_aucs.max())
        trained_model = batched_nn_train(train_x_sample, train_y_sample, model, 100, best_para[0], best_para[1]) # training model
        y_pred,_ = nn_test(test_x_sample, test_y_sample, trained_model)
        nn_aucs[count] = roc_auc_score(test_y_sample,y_pred)
    return nn_aucs, reg_aucs, tree_aucs