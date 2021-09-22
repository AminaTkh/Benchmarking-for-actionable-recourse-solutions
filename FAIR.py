from fairsearchcore.models import FairScoreDoc
import fairsearchcore as fsc
import pandas as pd
import sys
import ast
from random import randint
import random
import time

def calculate_p_value(data):
	return data.astype(int).mean()
	



def fair_algorithm(input_fn, output_fn):
	#index of sensitive attribute SINGLE  
	ind = '1' 
	
	k = 400  # number of topK elements returned (value should be between 10 and 400)
	p = 0 # proportion of protected candidates in the topK elements (value should be between 0.02 and 0.98) 
	alpha = [0.007, 0.01,  0.1,  0.15] # significance level (value should be between 0.01 and 0.15)
	
	data = pd.read_csv(input_fn) 
	
	
	if input_fn.find('german') != -1:
		p = 0.41975308641975306
		print("it is german")
	else:
		p = 0.5740740740740741
		print("it is credit")
		
		
	data = data.drop(columns = ['data_values','flipset_values'])
	
	
	if len(data) < k:
		k = len(data)
		range_ = 1
	else:
		range_ = 50

	dftime = pd.DataFrame(columns = ['time'])	
	for j in range(range_):
		random.seed(j*7)	
		print("J : ", j*7)
		mas = random.sample(range(len(data)), k)	
		mas.sort(reverse = True )
		#print('MAS', mas)
		unfair_ranking = []	
		for i in range(k):	
			unfair_ranking.append(FairScoreDoc(data.iloc[mas[i], 1], data.iloc[mas[i], 3], not bool(data.iloc[mas[i], 2])))
		
		for al in alpha:
			fair = fsc.Fair(k, p, al)
		
			print(fair.is_fair(unfair_ranking))
			start_time = time.time()
			re_ranked = fair.re_rank(unfair_ranking)
			dftime = dftime.append({'time' :time.time() - start_time}, ignore_index = True)
			rez_fn = output_fn + str(al) + str(j)  +'.csv'
			with open(rez_fn, 'w') as mf:
				mf.write("datapoint,attr_val,score\n")
			rez_file = open(rez_fn, 'a')
			re_ranked.reverse()
			for x in re_ranked:
				rez_fline = str(x.id)+","+str(int(not x.is_protected))+","+str(x.score)+"\n"
				rez_file.write(rez_fline)
			rez_file.close()
			print(fair.is_fair(re_ranked))
	timefile = input_fn
	timefile = timefile.replace('initialrankings/', 'time/')
	timefile = timefile.replace('.csv', '')
	#print(timefile)
	dftime.to_csv(timefile + 'FAIR.csv')


