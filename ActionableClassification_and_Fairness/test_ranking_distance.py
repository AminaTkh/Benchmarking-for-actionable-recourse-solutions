import pandas as pd
from copy import copy
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from aa import UNACTIONABLE
from action_fairness_algorithm import ActionFairnessAlgorithm, FailedToImproveDataPointException
import sys
import ast

class DataPoint:

    def __init__(self, data_row_index, attr_val, data_values, flipset_values, score):
        self.index = data_row_index  
        self.current_values =  (ast.literal_eval(data_values))[:-1]
        self.flipset_values =  (ast.literal_eval(flipset_values))[:-1]
        self.attr_value = attr_val
        self.score = score


def rank_by_distance(input_fn, output_fn):
	 
	if input_fn.find('german') != -1:
		file = "dataset/german2_reduced.csv"
		ex = "germancost1.xlsx"	
	else:
		file = "dataset/credit_processed_small_equal.csv"
		ex = "creditcost1.xlsx"
	
	if  input_fn.find("MACE") != -1:
		ex = "mace" + ex
		#print("MACE")
	ex = "cost/" + ex
		
	df = pd.read_csv(file)
	y = df.iloc[:, 0]  # classification is first column
	x = df.iloc[:, 1:]
	
	data_points = []
	data = pd.read_csv(input_fn)
	#print(len(data))
	
	for i in range(len(data)):
		data_points.append(DataPoint(data.iloc[i,1], data.iloc[i,2],data.iloc[i,3], data.iloc[i,4],data.iloc[i,5] )) 
	
	#for p in data_points:
	#	print((p.score))
	
	costs = pd.read_excel(ex)
	costs = costs.iloc[:, 1].fillna(UNACTIONABLE).to_list()  # Only get costs, replace NaNs with UNACTIONABLE
	
	data_df = pd.DataFrame(x)  # a.x is ndarray
	
	
	#print("COSTS", costs)
	
	# Do Action Fairness on negative data points only
	b = ActionFairnessAlgorithm(clf = LogisticRegression(solver="liblinear"),
								costs=costs,
								data_points=data_points
								)
	testing_output = []
	dftime = pd.DataFrame(columns = ['time'])		
	#print(b.get_data_points())
	# Test results for a variety of proportional strictness and number of blocks
	for proportion_modifier in range(2,9):
		for number_of_blocks in range(1,8):
			# Time before
			
			start_time = time()
	
			b.run(data_df=data_df,
				  attr_col_index=1,
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
	
	#		new_x = x.drop(indices_to_drop)
	#
	#		for i, df_index in enumerate(indices_to_drop):
	#			new_x.loc[df_index] = new_values[i]
	#		new_x = new_x.sort_index()
			
		   # print(new_x)
		   # new_x.to_csv('resultrank/moluBoga.csv')
			final = pd.concat(b._get_final_data())
			print(len(final))
			final.to_csv(output_fn + str(number_of_blocks) + '.' + str(proportion_modifier) + '.csv')
	timefile = input_fn
	timefile = timefile.replace('baselinerankings/', 'time/')
	timefile = timefile.replace('.csv', '')
	#print(timefile)
	dftime.to_csv(timefile + 'AC.csv')

	
	
	print("Finished!")
