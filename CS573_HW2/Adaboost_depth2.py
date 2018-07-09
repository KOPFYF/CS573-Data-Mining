from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = [[i,j] for i in range(6) for j in range(3)]
Y = [0 for i in range(9)]+ [1 for i in range(9)]
Y[4]=1

Ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), 
	n_estimators=1, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
Ada.fit(X,Y)

iter = 1
while sum(Ada.predict(X)-Y) != 0:	
	iter += 1
	Ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), 
	n_estimators=iter, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
	Ada.fit(X,Y)	
print(iter)