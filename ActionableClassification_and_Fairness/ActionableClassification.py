import pandas as pd
from copy import copy
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from aa import UNACTIONABLE
from actionable_classification_algorithm import ActionableClassificationAlgorithm
from action_fairness_algorithmProb import ActionFairnessAlgorithm, FailedToImproveDataPointException
import sys



def runAC(dataset):
	if dataset == "german":
		file = "dataset/german2_reduced.csv"
		ex = "input_costs_german.xlsx"	
	else:
		file = "dataset/credit_processed_small_equal.csv"
		ex = "input_costs_credit.xlsx"
	
	ex = "cost/" + ex
	df = pd.read_csv(file)
	
	# Train the classifier
	y = df.iloc[:, 0]  # classification is first column
	x = df.iloc[:, 1:]
	
	# Get costs from excel file
	
	costs = pd.read_excel(ex)
	costs = costs.iloc[:x.shape[1], 1].fillna(UNACTIONABLE).to_list()  # Only get costs, replace NaNs with UNACTIONABLE
	
	
	attr_col_index = 1  # Index of sensitive attribute (not including 1st classification column)
	attr_names = df.columns.to_list()[1:]
	#print(attr_names)
	
	clf = LogisticRegression(solver="liblinear")
	clf.fit(x, y)
	
	# Get flipsets for all negatively classed rows
	a = ActionableClassificationAlgorithm(costs=costs,
										  clf=clf,
										  clf_type='LR',
										  output_file=dataset,
										  x=x,
										  y=y,
										  sensitive_attr_index=None,
										  proportion_strictness=None,
										  number_of_blocks=None,
										  attr_names=attr_names,
										  )
	a.run()
	