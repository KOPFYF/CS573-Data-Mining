import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def plot_AUC(train_frac, AUC_result, type):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.plot(train_frac, AUC_result)
    plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
               [r'$10%$', r'$30%$', r'$50%$', r'$70%$', r'$90%$', r'$100%$'])
    plt.xlabel('training examples fraction')
    plt.ylabel('AUC' + type)
    plt.title(type)
    plt.show()
    fig.savefig('AUC' + type + '.png')


df = pd.read_csv("test.csv")
test_x = df.drop(['Loan ID', 'Status (Fully Paid=1, Not Paid=0)'], axis=1)
test_x = test_x.values
df = pd.read_csv("pred.csv")
test_y = df.drop(['Loan ID'], axis=1)
test_x[np.isnan(test_x)] = 0

AUC_xgb_result = []
AUC_logreg_result = []

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

    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300,
                            learning_rate=0.05).fit(train_x, train_y)
    pred_y_xgb = gbm.predict(test_x)
    AUC_xgb = roc_auc_score(test_y, pred_y_xgb)

    logreg = LogisticRegression(penalty='l2').fit(train_x, train_y)
    pred_y_logreg = logreg.predict(test_x)
    AUC_logreg = roc_auc_score(test_y, pred_y_logreg)

    AUC_xgb_result.append(AUC_xgb)
    AUC_logreg_result.append(AUC_logreg)

plot_AUC(train_frac, AUC_xgb_result, 'XGBoost')
plot_AUC(train_frac, AUC_logreg_result, 'Logistic Regression')

thefile = open('XGB_AUC_result.txt', 'w')
for item in AUC_xgb_result:
    thefile.write("%s\n" % item)

thefile = open('LOG_AUC_result.txt', 'w')
for item in AUC_logreg_result:
    thefile.write("%s\n" % item)
