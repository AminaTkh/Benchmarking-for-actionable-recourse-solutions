import ast
import numpy as np
from statistics import mean
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from math import sqrt
import pandas as pd
import sys


def measureCEmetrics (input_fn):
	file = open(input_fn, "r") 
	contents = file.read()
	file.close()
	
	d = ast.literal_eval(contents)
	#Sparsity means the average amount of features changed in counterfactuals
	time, distances, sparsity = [], [], []
	x_test, y_test = [], []
	for key in d.keys():
		dd = d[key]
		time.append(dd['cfe_time'])
		distances.append(dd['cfe_distance'])
		sample = dd['cfe_sample']
		init_sample = dd['fac_sample']
		test_exm = list(sample.values())
		x_test.append(test_exm[:-1])
		y_test.append(test_exm[-1])
		len_of_sample = len(sample)
		sparsity.append(len_of_sample - 1 - len(sample.items() & init_sample.items()))
	
	print(input_fn)
	print("The average time is ", mean(time))
	print("The average distance is ", mean(distances))
	print("The sparsity (average amount of features changed) is ", mean(sparsity))
	
	
	#MACE
	if input_fn.find('german') != -1  and  input_fn.find('MACE') != -1 :
		filename = "dataset/germanmace.csv"
		labelname = 'GoodCustomer (label)'
	if input_fn.find('credit') != -1  and  input_fn.find('MACE') != -1:
		filename = "dataset/creditmace.csv"
		labelname = 'NoDefaultNextMonth (label)'
	#AC and Rec
	if input_fn.find('german') != -1  and  input_fn.find('MACE') == -1 :
		filename = "dataset/german2_reduced.csv"
		labelname = 'GoodCustomer'
	if input_fn.find('credit') != -1  and  input_fn.find('MACE') == -1:
		filename = "dataset/credit_processed_small_equal.csv"
		labelname = 'NoDefaultNextMonth'
	
	
	df = pd.read_csv(filename)
	y_train = df[labelname]
	df = df.drop(labelname, axis=1)
	x_train = df
	
	from sklearn.neighbors import NearestNeighbors
	import numpy as np
	
	for i in range (1,2):
		nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(x_train)
		distances, indices = nbrs.kneighbors(x_test)
		print("The closeness to the training data is ", mean(distances[:,0]))
	
	"""
	rmse_val = [] #to store rmse values for different k
	for K in range(7):
		K = K+1
		model = neighbors.KNeighborsRegressor(n_neighbors = K)
		model.fit(x_train, y_train)  #fit the model
		pred = model.predict(x_test) #make prediction on test set
		acc = accuracy_score(y_test, pred.round())
		print("Accuracy value for k = ", K, " is", acc)
		error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
		rmse_val.append(error) #store rmse values
		print('RMSE value for k= ' , K , 'is:', error)
		"""
