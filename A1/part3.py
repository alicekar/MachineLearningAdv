from scipy.spatial import distance
import itertools as it
from math import exp, sqrt, pi 
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def generateData():
	t_values = list(it.product([-1,1], repeat = 9)) 
	data_sets = []
	for i in range(len(t_values)):
		element = t_values[i]
		element = np.asarray(element)
		grid = np.reshape(element,(3,3))
		data_sets.append(grid)
	# in the text the locations are 
	# [[(-1,1),(0,1),(1,1)],[(-1,0),(0,0),(1,0)],[(-1,-1),(0,-1),(1,-1)]]
	# but here it corresponds to
	# [[(0,0),(0,1),(0,2)],[(1,0),(1,1),(1,2)],[(2,0),(2,1),(2,2)]]
	# print(data_sets[9][1][2])
	return(data_sets)


def visualise(dataset):
	# OBS, takes a single data set as argument
	for i in range(3):
		print(dataset[i])


def plotData(dataset):
	if dataset[0][0]==1:
		plt.plot(-1,1, 'rs')
	else:
		plt.plot(-1,1, 'gs')
	if dataset[0][1]==1:
		plt.plot(0,1, 'rs')
	else:
		plt.plot(0,1, 'gs')
	if dataset[0][2]==1:
		plt.plot(1,1, 'rs')
	else:
		plt.plot(1,1, 'gs')


	if dataset[1][0]==1:
		plt.plot(-1,0, 'rs')
	else:
		plt.plot(-1,0, 'gs')
	if dataset[1][1]==1:
		plt.plot(0,0, 'rs')
	else:
		plt.plot(0,0, 'gs')
	if dataset[1][2]==1:
		plt.plot(1,0, 'rs')
	else:
		plt.plot(1,0, 'gs')


	if dataset[2][0]==1:
		plt.plot(-1,-1, 'rs')
	else:
		plt.plot(-1,-1, 'gs')
	if dataset[2][1]==1:
		plt.plot(0,-1, 'rs')
	else:
		plt.plot(0,-1, 'gs')
	if dataset[2][2]==1:
		plt.plot(1,-1, 'rs')
	else:
		plt.plot(1,-1, 'gs')

	#plt.show()


def generateModel(model, dataset, parameters):
	if model==0:
		return(1/512)
	p=1
	for i in range(3):             # x1 = rows
		for j in range(3):         # x2 = columns
			if model==1:		   # products
				p = p*1/(1+np.exp(-dataset[i,j]*parameters[0]*(i-1)))
			if model==2:
				p = p*1/(1+np.exp(-dataset[i,j]*(parameters[0]*(i-1)+parameters[1]*(j-1))))
			if model==3:
				p = p*1/(1+np.exp(-dataset[i,j]*(parameters[0]*(i-1)+parameters[1]*(j-1)+parameters[2])))
	return(p)


def samplePrior(model, samples):
	mu =  np.ones(model)           #np.zeros(model)
	sigma2 = 10
	Sigma = np.eye(model)*sigma2
	prior = np.random.multivariate_normal(mu, Sigma, samples)
	return(prior)  # returns a bunch of parameter "guesses"


def Evidence(model, dataset, samples):   
	p = 0
	for i in range(len(samples)):     # sum
		p = p + generateModel(model, dataset, samples[i])    # samples = parameters
	S = len(samples)
	evidence = p/S
	return(evidence)



def create_index_set(evidence):
    E = evidence.sum(axis=1)
    # change 'euclidean' to 'cityblock' for manhattan distance
    dist = distance.squareform(distance.pdist(evidence, 'euclidean'))
    np.fill_diagonal(dist, np.inf)
    
    L = []
    D = list(range(E.shape[0]))
    L.append(E.argmin())
    D.remove(L[-1])
    
    while len(D) > 0:
        # add d if dist from d to all other points in D
        # is larger than dist from d to L[-1]
        N = [d for d in D if dist[d, D].min() > dist[d, L[-1]]]
        
        if len(N) == 0:
            L.append(D[dist[L[-1],D].argmin()])
        else:
            L.append(N[dist[L[-1],N].argmax()])
        
        D.remove(L[-1])
    
    # reverse the resulting index array
    return np.array(L)[::-1]





#drawDataset(l[0])
S = 100
samples1 = samplePrior(1,S)
samples2 = samplePrior(2,S)
samples3 = samplePrior(3,S)
l = generateData()
#print(l[9])
#plotData(l[9])

evidence = np.zeros([4,512])

for i in range(4):
    for j in range(512):
        if i == 0:
            evidence[i][j]=Evidence(i, l[j], samples1)
        if i == 1:
            evidence[i][j]=Evidence(i, l[j], samples1)
        if i == 2:
            evidence[i][j]=Evidence(i, l[j], samples2)
        if i == 3:
            evidence[i][j]=Evidence(i, l[j], samples3)


max = np.argmax(evidence,axis=1)
min = np.argmin(evidence,axis=1)
sum = np.sum(evidence, axis=1)



index = create_index_set(evidence.T)
plt.plot(evidence[0,index],'pink', linestyle='--', label = "P(${D}$|${M}_0$)")
plt.plot(evidence[1,index],'b', label= "p(${D}$|${M}_1$)")
plt.plot(evidence[2,index],'r', label= "p(${D}$|${M}_2$)")
plt.plot(evidence[3,index],'g', label= "p(${D}$|${M}_3$)")
plt.legend()
plt.axis([-1,512, 0,0.15])
plt.ylabel('Model Evidence p(${D}$|${M}_i$)')
plt.xlabel('All possible data sets, D')
plt.show()


plt.plot(evidence[0,index],'pink', linestyle='--', label = "P(${D}$|${M}_0$)")
plt.plot(evidence[1,index],'b', label= "p(${D}$|${M}_1$)")
plt.plot(evidence[2,index],'r', label= "p(${D}$|${M}_2$)")
plt.plot(evidence[3,index],'g', label= "p(${D}$|${M}_3$)")
plt.legend()
plt.axis([-1,90, 0,0.15])
plt.ylabel('Evidence, p(${D}$|${M}_i$)')
plt.xlabel('Subset of all possible data sets, D')
plt.show()
'''

for i in range(len(max)):
	print(l[max[i]])
	plt.title('Most probable part of D for model '+str(i))
	plotData(l[max[i]])
	plt.show()

	print(l[min[i]])
	plt.title('Least probable part of D for model '+str(i))
	plotData(l[min[i]])
	plt.show()

print(max)
print(min)
print(sum)

'''


'''
datasets = generateData()
dataset = datasets[9]
#visualise(dataset)
#print(datasets)
#print(dataset)
#M = generateModel(3,dataset,[2,4,1])
#print(M)
S = samplePrior(2, 10)
#print(Evidence(1,dataset,S))
print(len(S))
print(S)
#for i in range(len(l)):
#	print(Evidence(2,l[i],S))

'''


