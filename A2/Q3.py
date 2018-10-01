# Author: Alice Karnsund
import numpy as np


# Create a array of the K tables visited for player n
def tableRoute(K, startProb):
	tables = np.zeros(K)
	# Choosing first table group
	start = np.random.binomial(1, startProb, 1)
	tables[0] = start
	for i in range(tables.shape[0]-1):
		if tables[i] == 1:
			following = np.random.binomial(1, 1/4, 1)
		else:
			following = np.random.binomial(1, 3/4, 1)
		tables[i+1] = following

	# Checking when the player is at either table group
	k1 = np.where(tables == 0)[0]
	k2 = np.where(tables == 1)[0]
	return(tables, k1, k2)


# Create an array that returns the observations from all tables
def diceOutcome(tables, k1, k2, distr1, distr2):
	# Results from each table group
	results1 = np.zeros(len(k1))
	results2 = np.zeros(len(k2))
	# All results
	results = np.zeros(len(tables))
	i = 0
	j = 0
	for k in range(len(tables)):
		if tables[k] == 0:
			# Throws each tables dice once
			prob = np.random.multinomial(1, distr1[k,:])
			results1[i] = np.where(prob == 1)[0]+1
			i+=1	
		else:
			prob = np.random.multinomial(1, distr2[k,:])
			results2[j] = np.where(prob == 1)[0]+1
			j+=1

		results[k] = np.where(prob == 1)[0]+1
	return(results1, results2, results)


# different categorical distributions
def catDist(K):
	# Biased dice in a random way, different at each table
	catTables1 = np.zeros((K, 6))
	catTables2 = np.zeros((K, 6))
	for k in range(K):
		rand1 = np.random.rand(1,6)[0]
		s1 = sum(rand1)
		rand1 = rand1/s1
		catTables1[k]=rand1

		rand2 = np.random.rand(1,6)[0]
		s2 = sum(rand2)
		rand2 = rand2/s2
		catTables2[k]=rand2

	# Fair dice for every table in table group
	catEqual = np.ones((K,6))/6

	# Every other table in a group has a biased 
	# and every other has a fair dice
	catMix = np.ones((K,6))/6
	for k in range(0,K,2):
		catMix[k] = catTables1[k]

	# Extremely biased dice for every table in table group. 
	# Dice will only give 6
	cat6 = np.zeros((K,6)) 
	cat6[...,5] = np.ones(K)

	# Extremely biased dice for every table in table group. 
	# Dice will only give 1
	cat1 = np.zeros((K,6)) 
	cat1[...,0] = np.ones(K)

	return(catTables1, catTables2, catEqual, catMix, cat6, cat1)



def observations(K, results, p):
	sum_n = sum(results)
	X_n = results
	observations = np.zeros(K)
	# prob = 1 indicates that the outcome is observed, 
	# and prob = 0 means that it is hidden
	for k in range(K):
		prob = np.random.binomial(1, p, 1)
		obs = prob*X_n[k]
		observations[k] = obs
	return(observations, sum_n)


def allPlayers(N, K, p):
	# Array with the outcome sum for each player
	allSums = np.zeros(N)
	# Matrix with the observed outcome sequence 
	# for each player (row)
	allSeq = np.zeros((N,K))
	# All "hidden" and "un-hidden" outcomes for 
	# each player (row), just to check!
	allResults = np.zeros((N,K))
	for n in range(N):
		tables, k1, k2 = tableRoute(K, 0.5)
		distr1, distr2, catE, catM, cat6, cat1 = catDist(K)
		r1, r2, results = diceOutcome(tables, k1, k2, catM, catM)
		playerObs, playerSum = observations(K, results, p)
		allSums[n] = playerSum
		allSeq[n] = playerObs
		allResults[n] = results
	print()
	print('All sums')
	print(allSums)
	print('All hidden and unhidden outcomes')
	print(allResults)
	print('All observed outcomes')
	print(allSeq)
	return(allSums,allSeq)


# Choose how many players in Casio
N = 6
# Choose how many tables each player will visit
K = 10
# Choose probability for observing a dice outcome
p = 0.7
allPlayers(N, K, p)



