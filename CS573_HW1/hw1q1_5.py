# hw1q1_5.py
import pandas as pd

train_df = pd.read_csv('creditcard.csv')
grouped = train_df.groupby(train_df['Class'])
class0 = grouped.get_group(0) # 284315
class1 = grouped.get_group(1) # 492
num0, num1 = class0.shape[0],class1.shape[0]
print(num0, num1)
weight = {0:(num0+num1)/num0,1:(num0+num1)/num1}
X_train = train_df.drop(['Class', 'Time'], axis=1)
Y_train = train_df["Class"] 

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# Logistic Regression
logreg = LogisticRegression(penalty='l2',C=1e-8,class_weight = weight) 
logreg.fit(X_train, Y_train)
pred_train = logreg.predict(X_train)
# print(round(logreg.score(X_train, Y_train) * 100, 2)) # print score

CM = confusion_matrix(Y_train, pred_train)
print("Accuracy of positive class:",CM[1,1]/(CM[1,0]+CM[1,1]))
print("Accuracy of negative class:",CM[0,0]/(CM[0,0]+CM[0,1])) 