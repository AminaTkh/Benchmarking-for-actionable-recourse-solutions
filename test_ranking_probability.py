import pandas as pd
from copy import copy
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from aa import UNACTIONABLE
from actionable_classification_algorithm import ActionableClassificationAlgorithm
from action_fairness_algorithmProb import ActionFairnessAlgorithm, FailedToImproveDataPointException
import sys



def rank_by_probability(input_fn, output_fn):
	if input_fn.find('german') != -1:
		file = "dataset/german2_reduced.csv"
		ex = "input_costs_german.xlsx"	
		d_name = "german"
	else:
		file = "dataset/credit_processed_small_equal.csv"
		ex = "input_costs_credit.xlsx"
		d_name = "credit"
	
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
	print(attr_names)
	
	clf = LogisticRegression(solver="liblinear")
	clf.fit(x, y)
	
	# Get flipsets for all negatively classed rows
	a = ActionableClassificationAlgorithm(costs=costs,
										  clf=clf,
										  clf_type='LR',
										  output_file=None,
										  x=x,
										  y=y,
										  sensitive_attr_index=None,
										  proportion_strictness=None,
										  number_of_blocks=None,
										  attr_names=attr_names,
										  )
	a.run()
	
	# Get results from Actionable Classification (without doing Action Fairness)
	clf = a.clf
	costs = a.costs
	data_points = a.negative_data_points
	data_df = pd.DataFrame(a.x)  # a.x is ndarray
	print("COSTS", costs)
	print(len(data_points))

	# Do Action Fairness on negative data points only
	b = ActionFairnessAlgorithm(clf=clf,
								costs=costs,
								data_points=data_points
								)
	b._get_attr_values(1)
	ranked_list = b._create_ranked_list()
	print(ranked_list)
	dftime = pd.DataFrame(columns = ['time'])		
	initialdf = b._store_data_point_values_as_df(ranked_list)	
	initialdf.to_csv("initialrankings/" + d_name + "ACprob.csv")
	# Test results for a variety of proportional strictness and number of blocks
	for proportion_modifier in range(2,9):
		for number_of_blocks in range(1,8):
			# Time before
			start_time = time()	
			b.run(data_df=data_df,
				  attr_col_index=attr_col_index,
				  number_of_blocks=number_of_blocks*5,
				  proportion_strictness=proportion_modifier
				  )
	
			# Time after
			dftime = dftime.append({'time' :time() - start_time}, ignore_index = True)
			# Number of unfair blocks initially and after
			unfair_blocks_before = len(b.initially_unfair)
			unfair_blocks_after = len(b.unfair_blocks)
	
			# Collapse b.new_list
			ranked_list = [datapoint for block in b.new_list for datapoint in block]
			# Get modified values
			indices_to_drop = [point.index for point in ranked_list]
			new_values = [point.current_values for point in ranked_list]
			new_x = x.drop(indices_to_drop)
	
			for i, df_index in enumerate(indices_to_drop):
				new_x.loc[df_index] = new_values[i]
			new_x = new_x.sort_index()
			final = pd.concat(b._get_final_data())
			final.to_csv(output_fn + str(number_of_blocks) + '.' + str(proportion_modifier) + '.csv')
	timefile = input_fn
	timefile = timefile.replace('initialrankings/', 'time/')
	timefile = timefile.replace('.csv', '')
	#print(timefile)
	dftime.to_csv(timefile + 'AC.csv')


	
	print("Finished!")
	

