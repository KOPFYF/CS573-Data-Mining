def get_oof(clf, x_train, y_train, x_test):
 oof_train = np.zeros((ntrain,))  
 oof_test = np.zeros((ntest,))
 oof_test_skf = np.empty((NFOLDS, ntest))  #NFOLDS行，ntest列的二维array
 for i, (train_index, test_index) in enumerate(kf): #循环NFOLDS次
     x_tr = x_train[train_index]
     y_tr = y_train[train_index]
     x_te = x_train[test_index]
     clf.fit(x_tr, y_tr)
     oof_train[test_index] = clf.predict(x_te)
     oof_test_skf[i, :] = clf.predict(x_test)  #固定行填充，循环一次，填充一行
 oof_test[:] = oof_test_skf.mean(axis=0)  #axis=0,按列求平均，最后保留一行
 return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)  #转置，从一行变为一列