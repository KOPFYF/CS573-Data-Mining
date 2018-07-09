# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('creditcard.csv')
# print(train_df.head(5))
# print(train_df.tail(5))
# print(train_df.info())
# print(train_df.describe())
X_train = train_df.drop(['Class', 'Time'], axis=1)
Y_train = train_df["Class"] 

# Q1_5, resample data to make it balanced
# grouped = train_df.groupby(train_df['Class'])
# class0 = grouped.get_group(0) # 284315
# class1 = grouped.get_group(1) # 492
# num0, num1 = class0.shape[0],class1.shape[0]
# # class_weight = num1/num0
# class0 = class0.sample(n = num1)
# newData = pd.merge(class0,class1,how='outer')
# X_train = newData.drop(['Class', 'Time'], axis=1)
# Y_train = newData["Class"] 

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# Logistic Regression

# logreg = LogisticRegression(penalty='l2',C=1e-8) # C: Inverse of regularization strength, smaller values specify stronger regularization
logreg = LogisticRegression(penalty='l2',C=1e-8,class_weight = "balanced")
logreg.fit(X_train, Y_train)
pred_train = logreg.predict(X_train)
print(round(logreg.score(X_train, Y_train) * 100, 2)) # print score

CM = confusion_matrix(Y_train, pred_train)
print("Accuracy of positive class:",CM[1,1]/(CM[1,0]+CM[1,1])) # true positives, false negatives
print("Accuracy of negative class:",CM[0,0]/(CM[0,0]+CM[0,1])) #  true negatives