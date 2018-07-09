# predict the real GPS data of a flight given its noisy data #
#################       Yifan Fei            ################# 
#################       03/20/2018           ################# 

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import *
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv('training_data.csv')
# train_df = train_df.dropna(axis = 0, how='any') // drop NaN
train_df = train_df.interpolate() 
test_df = pd.read_csv('flight_testFeatures_toPredict.csv')
print(train_df.head(20))
train_df.info()
ID_columns = ['Id', 'SegmentId_Time']
feature_columns = ['Noisy_Altitude','Noisy_Latitude','Noisy_Longitude']
prediction_columns = ['Altitude','Latitude','Longitude']
X_train = train_df[feature_columns]
Y_train = train_df[prediction_columns]
X_test  = test_df[feature_columns]

# print(X_train.head(10))
# print(Y_train.head(10))
# print(X_test.head(10))

# Decision Tree

# decision_tree = DecisionTreeRegressor(max_depth=3)
# decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
# print(Y_pred[0:100])
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print(acc_decision_tree)

# Random Forest
# http://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html

# random_forest_Altitude = RandomForestRegressor(max_depth=2, n_estimators=100)
# random_forest_Altitude.fit(X_train, Y_train['Altitude'])
# Y_pred_Altitude = random_forest_Altitude.predict(X_test)

# random_forest_Latitude = RandomForestRegressor(max_depth=2, n_estimators=100)
# random_forest_Latitude.fit(X_train, Y_train['Latitude'])
# Y_pred_Latitude = random_forest_Latitude.predict(X_test)

# random_forest_Longitude = RandomForestRegressor(max_depth=2, n_estimators=100)
# random_forest_Longitude.fit(X_train, Y_train['Longitude'])
# Y_pred_Longitude = random_forest_Longitude.predict(X_test)

# print(type(Y_pred_Longitude))
# print(Y_pred_Longitude[0:10])
# random_forest_Longitude.score(X_train, Y_train['Longitude'])
# acc_random_forest_Longitude = round(random_forest_Longitude.score(X_train, Y_train['Longitude']) * 100, 2)
# print(acc_random_forest_Longitude)

# submission = pd.DataFrame({
#         'SegmentId_Time': test_df["SegmentId_Time"],
#         'Altitude': Y_pred['Altitude'],
#         'Latitude': Y_pred['Latitude'],
#         'Longitude': Y_pred['Longitude']
#     })
# submission.to_csv('Submission.csv', index=False)