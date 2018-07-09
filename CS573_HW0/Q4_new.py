# Part a, 11 lines #
import numpy as np
import csv

seasons = {'winter': -1, 'summer':1, 'fall':0, 'spring':0}
List = []; # A flat list to store data
with open('matrix.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		List.append([int(row[0].strip()),int(row[1].strip()),seasons[row[2].strip()]])
	A = 999*np.ones((max(List,key = lambda x:x[0])[0]+1,max(List,key = lambda x:x[1])[1]+1))
	for row in reader:
		A[row[0],row[1]]=row[2] # convert List to matrix A
