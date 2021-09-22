
import numpy as np 
import pandas as pd
import sys
from cvxopt import spmatrix, matrix, sparse, solvers
from Birkhoff import birkhoff_von_neumann_decomposition
import random
import time
class Candidate(object):
  	def __init__(self, index, isProtected, score):
  		self.__index = index
  		self.__isProtected = isProtected
  		self.__score = score
  	
  	@property
  	def isProtected(self):
  		return self.__isProtected
  	
  	@property
  	def score(self):
  		return self.__score
  	
  	@property
  	def index(self):
  		return self.__index
  	
  	def set_index(self, value):
  		self.__index = value
 

def createRanking(x, nRanking, k):
    
    """
    Calculates the birkhoff-von-Neumann decomopsition using package available at
    https://github.com/jfinkels/birkhoff
    
    @param x: doubly stochastic matrix 
    @param nRanking: nRanking: List with candidate objects from the data set ordered color-blindly
    @param k: length of the ranking
    
    return the a list with candidate objects ordered according to the new ranking
    """
    
    #compute birkoff von neumann decomposition
    result = birkhoff_von_neumann_decomposition(x)
    
    theta = 0
    final = 0
    #choose permuation matrix with highest probability
    for coefficient, permutation_matrix in result:
        final += coefficient
        #print(coefficient)
        #print(permutation_matrix)
        if theta < coefficient:
            theta = coefficient
            ranking = permutation_matrix
    #get positions of each document 
    #print(ranking)       
    positions = np.nonzero(ranking)[1]
    
    #convert numpy array to iterable list
    positions = positions.tolist()
    #print(positions)
    top = nRanking[:k]
    tail = nRanking[k:]
    
    #sort top 40 scores according to index
  #  top.sort(key=lambda candidate: candidate.index, reverse=False)
    top.sort(key=lambda candidate: candidate.score, reverse=False)
    #make sure rest of ranking is still ordered color-blindly for evaluation with rKL
    tail.sort(key=lambda candidate: candidate.score, reverse=False)
    return nRanking, True

def solveLPWithDTC(ranking, k, dataSetName, algoName):
    
    """
    Solve the linear program with DTC
    
    @param ranking: list of candidate objects in the ranking
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    
    return doubly stochastic matrix as numpy array
    """
    
    print('Start building LP with DTC.')    
    #calculate the attention vector v using 1/log(1+indexOfRanking)
    u = []
    ranking.sort(key=lambda candidate: candidate.score, reverse=True)
    unproU = 0
    proU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.score)
    
    # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    arrayU = np.asarray(u)
    
    #normalize input
    arrayU = (arrayU - np.min(arrayU))/(np.max(arrayU)-np.min(arrayU))
    
    I = []
    J = []
    I2 = []
    #set up indices for column and row constraints
    for j in range(k**2):
        J.append(j)
    
    for i in range(k):
        for j in range(k):
            I.append(i)
            
    for i in range(k):
        for j in range(k):
            I2.append(j)
            
            
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proListX.append(i)
            proU += arrayU[i]
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += arrayU[i]
            
    arrayU = np.reshape(arrayU, (k,1))
    
    uv = arrayU.dot(v)
    uv = uv.flatten()
    
    #negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)
      
    # check if there are protected items    
    if proCount == 0:
        
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no protected items in data set.')
        return 0, False
    # check if there are unprotected items
    if unproCount == 0:
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
        
    proU = proU / proCount
    unproU = unproU / unproCount          
    
    initf = np.zeros((k,1))
    
    initf[proListX] = 1/(proCount*proU)
    initf[unproListX] = -(1/(unproCount*unproU))
    
    f1 = initf.dot(v)
    
    f1 = f1.flatten()
    f1 = np.reshape(f1, (1,k**2))
    
    f = matrix(f1)
         
    #set up constraints x <= 1
    A = spmatrix(1.0, range(k**2), range(k**2))
    #set up constraints x >= 0
    A1 = spmatrix(-1.0, range(k**2), range(k**2))
    #set up constraints that sum(rows)=1
    M = spmatrix(1.0, I,J)
    #set up constraints sum(columns)=1
    M1 = spmatrix(1.0, I2,J)
    #values for sums columns and rows == 1
    h1 = matrix(1.0, (k,1))
    #values for x<=1
    b = matrix(1.0, (k**2,1))
    #values for x >= 0
    d = matrix(0.0, (k**2,1))
    #construct objective function
    c = matrix(uv)
    #assemble constraint matrix as sparse matrix    
    G = sparse([M,M1,A,A1,f])
    
    #assemble constraint values
    h = matrix([h1,h1,b,d,0.0])
    
    print('Start solving LP with DTC.')
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + '.')
        return 0, False
    print('Finished solving LP with DTC.')
    
    return np.array(sol['x']), True

def solveLPWithDIC(ranking, k, dataSetName, algoName):
    
    """
    Solve the linear program with DIC
    
    @param ranking: list of candidate objects in the ranking
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    
    return doubly stochastic matrix as numpy array
    """
    
    print('Start building LP with DIC.')    
    #calculate the attention vector v using 1/log(1+indexOfRanking) 
    u = []
    ranking.sort(key=lambda candidate: candidate.score, reverse=True)
    unproU = 0
    proU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]
    
    for candidate in ranking[:k]:
        u.append(candidate.score)
    
    # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    arrayU = np.asarray(u)
    
    #normalize input
    arrayU = (arrayU - np.min(arrayU))/(np.max(arrayU)-np.min(arrayU))

    
    I = []
    J = []
    I2 = []
    #set up indices for column and row constraints
    for j in range(k**2):
        J.append(j)
    
    for i in range(k):
        for j in range(k):
            I.append(i)
            
    for i in range(k):
        for j in range(k):
            I2.append(j)
            
            
    for i in range(k):
        
        if ranking[i].isProtected == True:
            
            proCount += 1
            proListX.append(i)
            proU += arrayU[i]
            
        else:
            
            unproCount += 1
            unproListX.append(i)
            unproU += arrayU[i]
     
    arrayU = np.reshape(arrayU, (k,1))
    
    uv = arrayU.dot(v)
    uv = uv.flatten()
    
    #negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)    
    
    # check if there are protected items    
    if proCount == 0:
        
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no protected items in data set.')
        return 0, False
    # check if there are unprotected items
    if unproCount == 0:
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
        
    proU = proU / proCount
    unproU = unproU / unproCount          
    
    initf = np.zeros((k,1))
    
    initf[proListX] = (1/(proCount*proU))*arrayU[proListX]
    initf[unproListX] = (-(1/(unproCount*unproU))*arrayU[unproListX])
    
    f1 = initf.dot(v)
    
    f1 = f1.flatten()
    f1 = np.reshape(f1, (1,k**2))
    
    f = matrix(f1)
         
    #set up constraints x <= 1
    A = spmatrix(1.0, range(k**2), range(k**2))
    #set up constraints x >= 0
    A1 = spmatrix(-1.0, range(k**2), range(k**2))
    #set up constraints that sum(rows)=1
    M = spmatrix(1.0, I,J)
    #set up constraints sum(columns)=1
    M1 = spmatrix(1.0, I2,J)
    #values for sums columns and rows == 1
    h1 = matrix(1.0, (k,1))
    #values for x<=1
    b = matrix(1.0, (k**2,1))
    #values for x >= 0
    d = matrix(0.0, (k**2,1))
    #construct objective function
    c = matrix(uv)
    #assemble constraint matrix as sparse matrix    
    G = sparse([M,M1,A,A1,f])
    
    #assemble constraint values
    h = matrix([h1,h1,b,d,0.0])
    
    print('Start solving LP with DIC.')
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + '.')
        return 0, False
    print('Finished solving LP with DIC.')
    
    return np.array(sol['x']), True


def solveLPWithDPC(ranking, k, dataSetName, algoName):
    
    """
    Solve the linear program with DPC
    
    @param ranking: list of candidate objects in the ranking
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    
    return doubly stochastic matrix as numpy array
    """
    print('Start building LP with DPC.')
    u = []
  
    ranking.sort(key=lambda candidate: candidate.score, reverse=True)
    for candidate in ranking[:k]:
        u.append(candidate.score)

    #calculate the attention vector v using 1/log(1+indexOfRanking)
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX =[]

     # initialize v with DCG
    v = np.arange(1,(k+1),1)
    v = 1/np.log2(1 + v + 1)
    v = np.reshape(v, (1,k))
    
    arrayU = np.asarray(u)
    
    #normalize input
    arrayU = (arrayU - np.min(arrayU))/(np.max(arrayU)-np.min(arrayU))
    
    arrayU = np.reshape(arrayU, (k,1))
    
    uv = arrayU.dot(v)
    uv = uv.flatten()
    
    #negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)
    
    I = []
    J = []
    I2 = []
    #set up indices for column and row constraints
    for j in range(k**2):
        J.append(j)
    
    for i in range(k):
        for j in range(k):
            I.append(i)
            
    for i in range(k):
        for j in range(k):
            I2.append(j)
            
            
    for i in range(k):
        
        if ranking[i].isProtected == True: #isProtected???
            
            proCount += 1
            proListX.append(i)
            
        else:
            
            unproCount += 1
            unproListX.append(i)
        
    # check if there are protected items    
    if proCount == 0:
        
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no protected items in data set.')
        return 0, False
    # check if there are unprotected items
    if unproCount == 0:
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False
    
    initf = np.zeros((k,1))
    
    initf[proListX] = 1/proCount
    initf[unproListX] = -(1/unproCount)
    
    #build statistical parity constraint
    f1 = initf.dot(v)
    f1 = f1.flatten()
    f1 = np.reshape(f1, (1,k**2))
    f = matrix(f1)
         
    #set up constraints x <= 1
    A = spmatrix(1.0, range(k**2), range(k**2))
    #set up constraints x >= 0
    A1 = spmatrix(-1.0, range(k**2), range(k**2))
    #set up constraints that sum(rows)=1
    M = spmatrix(1.0, I,J)
    #set up constraints sum(columns)=1
    M1 = spmatrix(1.0, I2,J)
    #values for sums columns and rows == 1
    h1 = matrix(1.0, (k,1))
    #values for x<=1
    b = matrix(1.0, (k**2,1))
    #values for x >= 0
    d = matrix(0.0, (k**2,1))
    #construct objective function
    c = matrix(uv)
    #assemble constraint matrix as sparse matrix    
    G = sparse([M,M1,A,A1,f])
    #assemble constraint values
    
    h = matrix([h1,h1,b,d,0.0])
    
    print('Start solving LP with DPC.')
   
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        #print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + '.')
        return 0, False
    
    print('Finished solving LP with DPC.') 
    #print(sol)
    return np.array(sol['x']), True






def foeir_algorithm(input_fn, output_fn):
	data = pd.read_csv(input_fn) 
	data = data.drop(columns = ['data_values','flipset_values'])   
	k = 50
	range_ = 50
	dftime = pd.DataFrame(columns = ['time'])		
	#r = [i for i in range(51) if i != 9]	
	for j in range(range_):

	#for j in range(range_):
		random.seed(j*7)
		print("J : ", j*7)
		mas = random.sample(range(len(data)), k)	
		mas.sort(reverse = True )
		print("MAS", mas)
		unfair_ranking = []
		for i in range(len(mas)):
			c = Candidate(data.iloc[mas[i], 1], not bool(data.iloc[mas[i], 2]), data.iloc[mas[i], 3] )	
			unfair_ranking.append(c)
		
		metrics = ["DPC", "DIC", "DTC"]	
		for m in metrics:
			if m == "DPC":
				x, isRanked = solveLPWithDPC(unfair_ranking, k, input_fn, m)
			if m == "DIC":
				x, isRanked = solveLPWithDIC(unfair_ranking, k, input_fn, m)
			if m == "DTC":
				x, isRanked = solveLPWithDTC(unfair_ranking, k, input_fn, m)
							
			x = np.reshape(x,(k,k))
			x = np.asarray(x, dtype='float64')
			start_time = time.time()
			ranking, isRanked = createRanking(x, unfair_ranking, k)
			dftime = dftime.append({'time' :time.time() - start_time}, ignore_index = True)
			#print(time.time() - start_time)
			#print(dftime)
			rez_fn = output_fn + m + str(j) + '.csv'
			with open(rez_fn, 'w') as mf:
				mf.write(",datapoint,attr_val,score\n")
			rez_file = open(rez_fn, 'a')
			ranking.reverse()
			for x in ranking:
				rez_fline = "," + str(x.index)+","+str(int(not x.isProtected))+","+str(x.score)+"\n"
				rez_file.write(rez_fline)
			
			rez_file.close()
	timefile = input_fn
	timefile = timefile.replace('initialrankings/', 'time/')
	timefile = timefile.replace('.csv', '')
	#print(timefile)
	dftime.to_csv(timefile + 'FOEIR.csv')


