# Part a, 14 lines #
import numpy as np
import csv

def season2code(season):
	if "winter" in season: code = -1.  # strip max
	elif 'summer' in season: code = -1
	elif 'spring' or 'fall' in season: code = 0
	else: code = 999
	return code

A = 999*np.ones((200,200))
with open('matrix.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		(i,j) = (int(row[0]),int(row[1]))
		A[i,j]= season2code(row[2])

# Part b #
print(A[1:3,9:11])

# Part c #
(u,v) = (np.ones((1,10)),np.ones((1,10)))
print(np.inner(u, v))
