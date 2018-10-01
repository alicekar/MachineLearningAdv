import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.spatial.distance import cdist
from random import randint



def generateData():
	N=201
	Wtrue = np.array([1.5, -0.8])
	X = np.linspace(-2,2,N)
	errorVar = 0.2
	T = np.zeros(N) 
	Y = np.zeros(N) 
	for i in range(N):
		error = np.random.normal(0,errorVar)
		y = Wtrue[0]*X[i] + Wtrue[1]
		t = Wtrue[0]*X[i] + Wtrue[1] + error
		Y[i] = y
		T[i] = t
		i += 1
	#plt.plot(X[:], Y[:], 'black')	
	#plt.show()
	return(X,T)



def prior(): 
	mean = np.array([0,0])
	Covar =  np.identity(2)  # assuming sigma = 1
	'''
	alpha = 2   #precision  
	#Covar =  np.identity(2)/alpha  
	
	w0, w1 = np.random.multivariate_normal(mean, Covar, 5000000).T
	W = np.array((w0, w1)).T
	
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma', extent = [-2,2,-2,2])
	plt.title('Prior distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	plt.axis([-2, 2, -2, 2])
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show() 
	'''
	return(Covar)



def parameters(xIn, tIn, priorCov):
	ones = np.ones(len(xIn))
	x = np.array((xIn, ones)).T
	t = np.array(tIn).T
	Covar = np.linalg.inv(priorCov + np.dot(x.T, x))
	mean = np.dot(Covar, np.dot(x.T, t)).reshape(1,2)[0]
	return(mean, Covar)



def posterior(X, T):
	# 1 sample
	#x_sample = np.array([X[70]])
	#t_sample = np.array([T[70]])
	
	# 2 samples
	#x_sample = np.array([X[70], X[170]])
	#t_sample = np.array([T[70], T[170]])
	
	# 5 samples
	#x_sample = np.array([X[70], X[170], X[10], X[85], X[110]])
	#t_sample = np.array([T[70], T[170], T[10], T[85], T[110]])
	
	# 15 samples 
	#x_sample = np.array([X[70], X[170], X[50], X[85], X[110], X[103], X[120], X[90],
	#					 X[190], X[200], X[11], X[61], X[66], X[77], X[44]])
	#t_sample = np.array([T[70], T[170], T[50], T[85], T[110], T[103], T[120], T[90], 
	#				   	T[190], T[200], T[11], T[61], T[66], T[77], T[44]])
	
	# All points
	x_sample = X
	t_sample = T


	Samples = np.array((x_sample, t_sample)).T

	# Define new mean and covariance 
	priorCov = prior()
	mean, Covar = parameters(x_sample, t_sample, priorCov)
	
	# Plot posterior
	w0, w1 = np.random.multivariate_normal(mean, Covar, 5000000).T
	W = np.array((w0, w1)).T
	
	'''
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma', extent = [-2,2,-2,2])
	plt.title('Posterior distribution, '+ str(len(x_sample))+' data points observed')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	plt.axis([-2, 2, -2, 2])
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show() '''
	
	return(W, Samples)


def plotFunctions(X, W, Samples):
	# plot samples
	'''for i in range(len(Samples)):
		point = Samples[i]
		x = point[0]
		t = point[1]
		plt.plot(x,t,'bo')
		i += 1'''

	# functions with weights from posterior
	for i in range(5):
		r = randint(0,len(W))
		w = W[r]
		Y = X[:]*w[0] + w[1] 
		pb.plot(X[:], Y[:])
		i += 1

	plt.axis([-2, 2, -2, 2])
	plt.xlabel('x')
	plt.ylabel('y = $w_0$x + $w_1$')
	plt.title('data space')
	plt.show()	
	return()



	

X,T = generateData()
#prior()
W, Samples = posterior(X,T)
plotFunctions(X, W, Samples)



'''
# testade me L matris, dv Cov för likelihood

def parameters2(xIn, tIn, priorCov):
	ones = np.ones(len(xIn))
	x = np.array((xIn, ones)).T
	t = np.array(tIn).T
	CovarLH = np.eye(2)*0.2 
	priorPrecision = np.linalg.inv(priorCov)
	Covar = np.linalg.inv(priorPrecision + np.dot(x.T, np.dot(CovarLH, x)))
	mean = np.dot(Covar, np.dot(x.T,np.dot(CovarLH, t))).reshape(1,2)[0]
	return(mean, Covar)
'''







