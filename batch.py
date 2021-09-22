import sys, os
sys.path.extend([f'./{name}' for name in os.listdir(".") if os.path.isdir(name)])
from MeasuringFairness import *
from CreateRankingFromDictionary import *
from MeasureCGmetrics import *
import argparse
from FOEIR import *
from FAIR import *
from test_ranking_probability import *
from test_ranking_distance import *
from MeasureRatio import *
from statistics import mean
import pandas as pd


def get_dataset(str_):
	if str_ == 'german' or str_ == 'credit':
		return [str_]
	else:
		return ['german', 'credit']
	

def get_cg_method(str_):
	if str_ == 'AC' or str_ == 'AR' or str_ == 'MACE':
		return [str_]
	else:
		return ['AC', 'AR', 'MACE']
	

def get_rank(str_):
	if str_ == 'FAIR' or str_ == 'FOEIR' or str_ == 'AC' or str_ == 'INITIAL':
		return [str_]
	else:
		return ['FAIR', 'FOEIR', 'AC']
	

def get_base(str_):
	if str_ == 'dist' or str_ == 'prob':
		return str_
	else:
		return ['dist', 'prob']


def gettime(input_fn):
	df = pd.read_csv(input_fn)
	return df['time'].mean()


#FAIR FOEIR AC can rank MACE/AC/AR on distances
#FAIR FOEIR AC can rank AC on probabilities 


my_parser = argparse.ArgumentParser()
my_parser.version = '1.0'

# dataset name
my_parser.add_argument('-task') #MeasureCEmetrics, generateCE, CreateRankingFromDictionary, Ranking, MeasureFairness, GetTotalMetrics, GetTime
my_parser.add_argument('-d', default='all') #german, credit, both
my_parser.add_argument('-cg_method', default='all') #AC, AR, MACE
my_parser.add_argument('-base', default='all') #prob = probability, dist = distance
my_parser.add_argument('-rank', default='all') #FAIR, FOEIR, AC, INITIAL
args = my_parser.parse_args()


dataset = get_dataset(args.d)
cg_method = get_cg_method(args.cg_method)
rank = get_rank(args.rank)
base = get_base(args.base)
if args.task == "MeasureCEmetrics":
	for method in cg_method:
		for d in dataset:
			input_fn = "dictionaries/" + d + method + 'dist.txt'
			measureCEmetrics(input_fn)
			
if args.task == "MeasureFairness":
	data_folder = "rankings"
	if base == "dist":
		for method in cg_method:
			for r in rank:
				output_fn = "metrics/"   + method + base + r
				if r == 'INITIAL':
					get_metrics('initial' + data_folder, output_fn, dataset, method, base, rank = "")	
				else:
					get_metrics(data_folder, output_fn, dataset, method, base, r)	
				print ("Finished experiments on datasets")
				print ("Result stores in ", output_fn, ".csv")
	else:
	#base == prob
			for r in rank:
				output_fn = "metrics/" + "AC" + base + r
				if r == 'INITIAL':
					get_metrics('initial' + data_folder, output_fn, dataset, "AC", base, rank = "")	
				else:
					get_metrics(data_folder, output_fn, dataset,  "AC", base, r)	

				print ("Finished experiments on datasets")
				print ("Result stores in ", output_fn, ".csv")
		
	


if args.task == "CreateRankingFromDictionary":
	for d in dataset:
		for method in cg_method:
			input_fn = "dictionaries/" + d+method+'dist'+'.txt'
			output_fn = "initialrankings/" + d+method+'dist' + '.csv'
			create_ranking_from_dictionary(input_fn, output_fn)	
			print ("Finished creating ranking from dictionary")
			print ("Result stores in ", output_fn)

print(rank)

#Ranking FAIR FOEIR AC for all dist
#Ranking FAIR FOEIR AC for prob if cg_method = AC
if args.task == "Ranking":
	for d in dataset:
		for r in rank:	
			print(r)
			print(d)
			if base == "dist":
				for method in cg_method: 
					input_fn = "initialrankings/" + d + method + base + '.csv'
					output_fn = "rankings/" + r +'/' + d + method + base + r
					if r == "FOEIR":
						foeir_algorithm(input_fn, output_fn)
					if r == "FAIR":
						fair_algorithm(input_fn, output_fn)
					if r == "AC":
						print('hello')
						rank_by_distance(input_fn, output_fn)
			else:
			#base == prob
				input_fn = "initialrankings/" + d + "AC" + base + '.csv'
				output_fn = "rankings/" + r +'/' + d + "AC" + base + r
				if r == "FOEIR":
					foeir_algorithm(input_fn, output_fn)
				if r == "FAIR":
					fair_algorithm(input_fn, output_fn)
				if r == "AC":
					rank_by_probability(input_fn, output_fn)










def put_dict_to_csv(d1, d2, d3, st, filee):
	for x in d1.keys():
		filee.write(st + x + ',' + str(d1[x]) + ',' + str(d2[x]) +',' + str(d3[x]) + '\n')
		
	

#measure action fairness ratio : available only for AC cg and ranking (ACdistAC, ACprobAC, ACdist, ACprob)
if args.task == "GetTotalMetrics":
	rank.append('INITIAL')
	mfile = "metric.csv"
	with open(mfile, 'w') as mf:
		mf.write("Dataset,cg_method,base,rank,alpha,rKL,rND,rRD\n")
	m_str = open(mfile, 'a')
	
	
	ratiofile = "ratio_sp_kt.csv"
	with open(ratiofile, 'w') as rf:
		rf.write("Dataset,cg_method,base,rank,Block_size,Strictness,ratio,sp,kt\n")
	rf = open(ratiofile, 'a')

	base = 'dist'
	for method in cg_method:
		if (base == 'prob') and (method != 'AC'):
			continue			
		for r in rank:
			for d in dataset:
				print(method, ' ', r, ' ', d, ' ',base)
						
				if r == "FAIR" or r == "FOEIR":		
					#for dist				
					input_fn = 'metrics/' + method + base + r + '.csv'
					rKL, rND, rRD = getaverage (input_fn, d, r)
					put_dict_to_csv(rKL, rND, rRD, d + ',' + method + ',dist,' + r + ',', m_str)

					if method == "AC":
					#for prob				
						input_fn = 'metrics/' + method + 'prob' + r + '.csv'
						rKL, rND, rRD = getaverage (input_fn, d, r)
						put_dict_to_csv(rKL, rND, rRD, d + ',' + method + ',prob,' + r + ',', m_str)

					continue
					
				if r == "AC" and method == "AC":  #nothing for other cg methods (long files)
					input_fn = "rankings/AC/" + d + 'AC' + base + 'AC' 
					measureratio(input_fn, d + ',' + method + ',' + base + ',' + r + ',', rf)

					#prob
					input_fn = "rankings/AC/" + d + 'ACprobAC' 
					measureratio(input_fn, d + ',' + method + ',' + 'prob' + ',' + r + ',', rf)

					continue
					
					
				if r == "INITIAL":	
					input_fn = "metrics/" + method + base + r + '.csv'
					rKL, rND, rRD = getaverage (input_fn, d, r)
					put_dict_to_csv(rKL, rND, rRD, d + ',' + method + ',dist,' + r + ',', m_str)
					if method == "AC":
					#for prob				
						input_fn = 'metrics/' + method + 'prob' + r + '.csv'
						rKL, rND, rRD = getaverage (input_fn, d, r)
						put_dict_to_csv(rKL, rND, rRD, d + ',' + method + ',prob,' + r + ',', m_str)
						
						input_fn = "initialrankings/" + d + 'AC' + 'prob' + '.csv'
						measureratio(input_fn,  d + ',' + method + ',prob,' + r + ',', rf)	
						input_fn = "initialrankings/" + d + 'AC' + 'dist' + '.csv'
						measureratio(input_fn,  d + ',' + method + ',dist,' + r + ',', rf)	

					continue
					









if args.task == "GetTime":
	timefile = "time.csv"
	with open(timefile, 'w') as mf:
		mf.write("Dataset,cg_method,base,rank,avg_time\n")
	time_str = open(timefile, 'a')
	for d in dataset:
		for method in cg_method:
			for r in rank:
				input_fn = 'time/' + d + method + 'dist' + r + '.csv'
				time_str.write(d + ',' + method + ',dist,' + r + ',' + str(gettime(input_fn)) + '\n')
				if method == 'AC':
					input_fn = 'time/' + d + method + 'prob' + r + '.csv'
					time_str.write(d + ',' + method + ',prob,' +r + ',' + str(gettime(input_fn)) + '\n')
	print('Time computation is finished.')


					

		
			
			
