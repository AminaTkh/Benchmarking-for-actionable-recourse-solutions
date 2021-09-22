# Code based on package usage example from https://github.com/ustunb/actionable-recourse
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
from recourse.action_set import ActionSet
from recourse.flipset import Flipset
from scipy.spatial import distance
import sys
import ast




def get_AR_dict(dataset):
	
	if dataset == "german":
		url = "dataset/german2_reduced.csv"
		list_of_immutables = ['Gender','Single', 'Age'] #german
	else:
		url = "dataset/credit_processed_small_equal.csv"
		list_of_immutables = ['Married', 'Single']
	
	negative_outcome = -1 
	
	df = pd.read_csv(url)
	y, X = df.iloc[:, 0], df.iloc[:, 1:]
	
	X = X.iloc[:, 0:20]  # Can only take first 20 columns
	number_of_columns = len(X.iloc[0])
	
	# train a classifier
	clf = LogisticRegression(solver="liblinear")
	clf.fit(X, y)
	yhat = clf.predict(X)
	# customize the set of actions
	# matrix of features. ActionSet will learn default bounds and step-size.
	A = ActionSet(X)
	
	# Unmutable ('unactionable') attributes
	if dataset == "german":
		A['Gender'].mutable = False #german
		A['Age'].mutable = False #german
		A['Single'].mutable = False
	else:
		A['Married'].mutable = False  # credit
		A['Single'].mutable = False
	
	
	# Don't set bounds, as they set lb, ub from min max of data, as do we
	
	# Ensure it thinks values can be changed by one
	# Change values discretely instead of percentile for every attribute
	for i in range(number_of_columns):
		A[i].step_type = 'absolute'
		A[i].step_size = 1
	
	coefficients = clf.coef_[0]
	intercept = clf.intercept_[0]
	
	# get model coefficients and align
	# A.align(coefficients=coefficients)  ## tells `ActionSet` which directions each feature should move in to produce positive change
	
	# Get one individual
	# Only considering values where the model would predict them -1
	y_pred = clf.predict(X)
	# Get index of each predicted negative outcome
	negatives = [i for i, val in enumerate(y_pred) if val == negative_outcome]
	# For credit_processed, classification is on 'NoDefaultNextMonth'. Negative outcome is 1 (Default), positive is 0
	
	flipsets = []
	dict_for_avg = {}
	#leng = len(negatives)
	
	#if u want to get flipsets for all data change negatives to "range(len(X))"
	for i in negatives:
		start_time = time()
		x_prev = (X.iloc[i].to_list())
		f = Flipset(x=X.iloc[i].to_list(),
					action_set=A,
					clf=clf
					)
		time_taken = time() - start_time
		f.populate(enumeration_type='distinct_subsets',
				   
				   cost_type='total'
				   )
		df = f.view()
		df = df[df.cost == df.cost.min()]
		sample = {}
		x_new = x_prev[:]
		
		ind = list(df['feature_idx'][0])
		values = list(df['x_new'][0])
	
		
		for j in range(len(ind)):
			x_new[ind[j]] = values[j]
	
		indd = [str(j) for j in range(len(x_prev))]
		fac_sample = dict(zip(indd, x_prev))
		fac_sample['y'] = False
		
		indd = [str(j) for j in range(len(x_new))]
		cfe_sample = dict(zip(indd, x_new))
		cfe_sample['y'] = True
		#print(df)
		dst = distance.euclidean(x_prev, x_new)
		sample['cfe_distance'] = dst
		sample['cfe_time'] = time_taken
		sample['cfe_sample'] = cfe_sample
		sample['fac_sample'] = fac_sample
		#print(sample)
		dict_for_avg[str(i)] = sample
		flipsets.append(f)  # CHANGED to X.iloc[1], had to install jinja2
	
	
	with open("dictionaries/" + dataset + "AR.txt", 'w') as f:
		print(dict_for_avg, file=f)
	
	#print("Time taken was", time_taken )
	#print(len(flipsets))
