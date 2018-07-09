import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv("test.csv")
test_x = df.drop(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'], axis=1)
test_x = test_x.values
df = pd.read_csv("pred.csv")
test_y = df.drop(['Loan ID'], axis=1)
test_x[np.isnan(test_x)] = 0

df = pd.read_csv("train.csv")
train_x = df.drop(
    ['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'], axis=1)
train_y = df['Status (Fully Paid=1, Not Paid=0)']
train_x = train_x.values
train_y = train_y.values
train_x[np.isnan(train_x)] = 0
# print(train_x.shape, train_y.shape)

AUC_xgb_result = []
AUC_logreg_result = []

kfold = 10
skf = StratifiedKFold(n_splits=kfold, random_state=42)

for i, (train_index, test_index) in enumerate(skf.split(train_x,train_y)):
    
    train_X, test_X = train_x[train_index], train_x[test_index]
    train_Y, test_Y = train_y[train_index], train_y[test_index]

    logreg = LogisticRegression(penalty='l2').fit(train_X, train_Y)
    pred_y_logreg = logreg.predict(test_X)
    AUC_logreg = roc_auc_score(test_Y, pred_y_logreg)
    AUC_logreg_result.append(AUC_logreg)
    print('[Fold %d/%d Prediciton, AUC score: %s]' % (i + 1, kfold, AUC_logreg))
print('Mean and std AUC score for LogisticRegression:',np.asarray(AUC_logreg_result).mean(),np.asarray(AUC_logreg_result).std())
thefile = open('k_fold_AUC_result_logreg.txt', 'w')
for item in AUC_logreg_result:
	thefile.write("%s\n" % item)

# for i, (train_index, test_index) in enumerate(skf.split(train_x,train_y)):
    
#     train_X, test_X = train_x[train_index], train_x[test_index]
#     train_Y, test_Y = train_y[train_index], train_y[test_index]
#     gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300,
#                             learning_rate=0.05).fit(train_X, train_Y)
#     pred_y_xgb = gbm.predict(test_X)
#     AUC_xgb = roc_auc_score(test_Y, pred_y_xgb)
#     AUC_xgb_result.append(AUC_xgb)

#     print('[Fold %d/%d Prediciton, AUC score: %s]' % (i + 1, kfold, AUC_xgb))
# print('Mean AUC score for Xgboost:',np.asarray(AUC_xgb_result).mean(),np.asarray(AUC_xgb_result).std())

# thefile = open('k_fold_AUC_result_xgb.txt', 'w')
# for item in AUC_xgb_result:
# 	thefile.write("%s\n" % item) 
                        


