import ast
import numpy as np
from statistics import mean
from math import sqrt
import pandas as pd
import sys
from scipy import stats
#import distance
import math
from sklearn.metrics import ndcg_score

def costpergroup( df, costs, weighted = False):	
	costs_per_group = {}
	
	for idx, row in df.iterrows():
		value_changes = np.subtract(np.array(ast.literal_eval(row.data_values)), np.array(ast.literal_eval(row.flipset_values)))  # Get attr changes
		value_changes = np.power(value_changes, 2)  # Changes ** 2
		#print(value_changes)
		#print(costs)
		if weighted:
			value_changes = np.multiply(costs, value_changes)  # Weighted by costs
		value_changes = math.sqrt(np.sum(value_changes))  # Get total cost (sum)
	
		# Add to dict (of value sensitive attr val)
		current_costs = costs_per_group.get(row.attr_val, [])
		current_costs.append(value_changes)
		costs_per_group[row.attr_val] = current_costs

	return costs_per_group



def measureratio (input_fn, st, filee):
	if input_fn.find('german') != -1:
		ex = "input_costs_german.xlsx"	
		d_name = "german"
	else:
		ex = "input_costs_credit.xlsx"
		d_name = "credit"
	
	fileofinitrank = 'baselinerankings/' + d_name + 'AC'

	if input_fn.find('prob') != -1:
		fileofinitrank = fileofinitrank + 'prob.csv'
	else:
		fileofinitrank = fileofinitrank + 'dist.csv'
	print(fileofinitrank)
	init_d = pd.read_csv(fileofinitrank)
	ex = "cost/" + ex	
	costs = pd.read_excel(ex)

	if input_fn.find('baseline')!= -1 and input_fn.find('dist')!= -1:
	#simple case
		df = pd.read_csv(input_fn)
		value = df.groupby(['attr_val'])['score'].mean()
		#print(value)
		#print(value.min() / value.max())
		filee.write(st + 'init,init,' + str(value.min() / value.max()) + '\n')
		
		
	if input_fn.find('baseline')!= -1 and input_fn.find('prob')!= -1:
		df = pd.read_csv(input_fn)
		costs_per_group_after = costpergroup(df, costs.iloc[:, 1], True)
		mean_group_costs_after = {group: mean(
            costs) for group, costs in costs_per_group_after.items()}
		#print(mean_group_costs_after)
		#print(min(mean_group_costs_after.values())/max(mean_group_costs_after.values()))
		filee.write(st + 'init,init,' + str(min(mean_group_costs_after.values())/max(mean_group_costs_after.values())) + '\n')
		
	
	if input_fn.find('baseline') == -1:	
		for proportion_modifier in range(2,9):
			for number_of_blocks in range(1,8):
				df = pd.read_csv(input_fn+str(number_of_blocks) + '.' + str(proportion_modifier)+'.csv')
				costs_per_group_after = costpergroup(df, costs.iloc[:, 1], (input_fn.find('prob')!= -1))
				mean_group_costs_after = {group: mean(costs) for group, costs in costs_per_group_after.items()}
				#print(number_of_blocks, ' ',proportion_modifier,' ')
				#print(min(mean_group_costs_after.values())/max(mean_group_costs_after.values()))
				sp = stats.spearmanr(df['datapoint'], init_d['datapoint'])
				kt = stats.kendalltau(df['datapoint'], init_d['datapoint'])
				filee.write(st  + str(number_of_blocks*5) + ',' + str(proportion_modifier) + ',' + str(min(mean_group_costs_after.values())/max(mean_group_costs_after.values())) +  '\n')


def getaverage(input_fn, d, r):

	df = pd.read_csv(input_fn)
	#print(df)
	df = df[df['Dataset'] == d]
	
	if r == 'FOEIR':
		DPC = df.iloc[::3, :]
		DTC = df.iloc[1::3, :]
		DIC = df.iloc[2::3, :]
		valuesrKL = {'DPC' : DPC['rKL'].mean(), 'DTC' : DTC['rKL'].mean(), 'DIC' : DIC['rKL'].mean()}
		
		valuesrND = {'DPC' : DPC['rND'].mean(), 'DTC' : DTC['rND'].mean(), 'DIC' : DIC['rND'].mean()}
		
		valuesrRD = {'DPC' : DPC['rRD'].mean(), 'DTC' : DTC['rRD'].mean(), 'DIC' : DIC['rRD'].mean()}



	if r == 'FAIR':
		alpha0_007 = df.iloc[::4, :]
		alpha0_01 = df.iloc[1::4, :]
		alpha0_1 = df.iloc[2::4, :]
		alpha0_15 = df.iloc[3::4, :]
		valuesrKL = {'0.007' : alpha0_007['rKL'].mean(), '0.01' : alpha0_01['rKL'].mean(), '0.1' : alpha0_1['rKL'].mean(), '0.15' : alpha0_15['rKL'].mean()}
		
		valuesrND  = {'0.007' : alpha0_007['rND'].mean(), '0.01' : alpha0_01['rND'].mean(), '0.1' : alpha0_1['rND'].mean(), '0.15' : alpha0_15['rND'].mean()}

		valuesrRD  = {'0.007' : alpha0_007['rRD'].mean(), '0.01' : alpha0_01['rRD'].mean(), '0.1' : alpha0_1['rRD'].mean(), '0.15' : alpha0_15['rRD'].mean()}


	if r == 'BASELINE':
		valuesrKL = {'init' : df['rKL'].mean()}		
		valuesrND = {'init' :df['rND'].mean()}			
		valuesrRD = {'init' :df['rRD'].mean()}	



	print ('rKL : ', valuesrKL, '\n' )
	print ('rND : ', valuesrND, '\n'  )
	print ('rRD : ', valuesrRD, '\n'  )
	return valuesrKL, valuesrND, valuesrRD
