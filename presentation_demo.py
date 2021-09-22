import pandas as pd
from aa import actionable_classification, UNACTIONABLE
from action_fairness_algorithm import ActionFairnessAlgorithm
from chart import german_chart
from sklearn.linear_model import LogisticRegression
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import sys
import ast
# import data
if sys.argv[1] == "german":
	url = "german2_reduced.csv"
	costfile = "input_costs_german.xlsx"
	list_of_immutables = ['Gender','Single', 'Age'] #german
else:
	url = "credit_processed_small_equal.csv"
	costfile = "input_costs_credit.xlsx"
	list_of_immutables = ['Married', 'Single']
	
df = pd.read_csv(url)

# Train the classifier
y = df.iloc[:, 0]  # classification is first column
X = df.iloc[:, 1:]

# Get costs from excel file
costs = pd.read_excel(costfile)
costs = costs.iloc[:, 1].fillna(UNACTIONABLE).to_list()  # Only get costs, replace NaNs with UNACTIONABLE

clf = LogisticRegression(solver='liblinear')
clf.fit(X, y)

# Run Actionable Classification
AC = actionable_classification(df,
                               costs,
                               output_file='demo_test.csv',
                               clf=clf,
                               clf_type='LR',
                               action_fairness=False,  # So can walk through flipsets before looking into AF
                               sensitive_attr_index=1  # Gender
                             
                               )

filename = "demo_test.csv"
subprocess.Popen([filename], shell=True)

print("Flipsets Calculated")

