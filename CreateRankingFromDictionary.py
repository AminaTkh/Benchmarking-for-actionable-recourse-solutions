import ast
import numpy as np
from statistics import mean
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from math import sqrt
import pandas as pd
import sys


def create_ranking_from_dictionary(input_fn, output_fn):
	file = open(input_fn, "r") 
	contents = file.read()
	file.close()
	
	d = ast.literal_eval(contents)
	data_list = []
	
	for key in d.keys():
		dd = d[key]
		if key.isnumeric() == False:
			key = key.replace("sample_", "")
		init_sample = dd['fac_sample']
		sample_values = list(init_sample.values())
		cfe_sample = dd['cfe_sample']
		cfe_values = list(cfe_sample.values())
		data_list.append(
					{"datapoint": key,
					 "attr_val": sample_values[1],
					 "data_values": sample_values,
					 "flipset_values": cfe_values,
					 "score": dd['cfe_distance'],
					 })  # Get methods as reference value passed instead of actual value.
	
	data_df = pd.DataFrame(data_list)
	#data_df = data_df.sort_values(by = "predict_prob_val")
	data_df = pd.DataFrame(data_df.sort_values(["score"], ascending = True).to_numpy(), 
	
					   index=data_df.index, columns=data_df.columns)
	
	data_df.to_csv(output_fn)