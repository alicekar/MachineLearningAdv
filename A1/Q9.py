import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.spatial.distance import cdist
# To sample from a multivariate Gaussian
#f = np.random.multivariate_normal(mu,K);
# To compute a distance matrix between two sets of vectors
#D = cdist(x1,x2)
# To compute the exponetial of all elements in a matrix
#E = np.exp(D)

def generateData():
	N=201
	Wtrue = np.array([1.5, -0.8])
	X = np.linspace(-2,2,N)
	errorVar = 0.2
	T = np.zeros(N)
	for i in range(N):
		error = np.random.normal(0,errorVar)
		t = Wtrue[0]*X[i] + Wtrue[1] + error
		#plt.plot(X[i],t,'bo')
		T[i] = t
		i += 1
	#plt.show()
	return(X,T)

X,T = generateData()



def prior(): 
	mean = np.array([0,0])
	alpha = 2   #precision  
	Covar =  np.identity(2) #*0.2  #np.identity(2)/alpha 

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
	
	return(W)

	
#prior()

def posterior(X,T):
	sigma2 = 0.2   # Noise variance
	index = 100
	xp = X[index]
	tp = T[index]
	#xp = np.array([X[60],X[160]])
	#tp = np.array([T[60],T[160]])
	sample = [X[1], T[1]]
	cov_sample = np.identity(2) + np.array([[sample[0]*sample[0], sample[0]], [sample[0], 1]])
	cov_sample = np.linalg.inv(cov_sample)
	mean_sample = sample[0]*np.dot(cov_sample, np.array([[sample[0]],[1]])).reshape(1,2)
	print(mean_sample)
	print(mean_sample[0])
	alpha = 2   #precision
	Sigma =  np.identity(2)#*0.2  # np.identity(2)/alpha  # Covariance from prior
	SigmaInv = np.linalg.inv(Sigma)
	term1 = ((1/sigma2)*np.dot(xp,xp) + SigmaInv)
	mean = (1/sigma2)*np.linalg.inv(term1)*np.dot(xp,tp)
	print(mean)
	mean = mean[0]
	print(mean)

	term2 = Sigma    #(1/sigma2)*np.dot(xp,xp) + SigmaInv
	Covar = np.linalg.inv(term2)
	mean = mean_sample[0]
	Covar = cov_sample

	w0, w1 = np.random.multivariate_normal(mean, Covar, 500000).T
	W = np.array((w0, w1)).T
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma', extent = [-2,2,-2,2])
	plt.title('Prior distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	plt.axis([-2, 2, -2, 2])
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show()

posterior(X,T)


def posterior2(X,T):
	sigma2 = 0.2   # Noise variance
	alpha = 2   #precision
	Sigma =  np.identity(2)*0.2  # np.identity(2)/alpha  # Covariance from prior
	SigmaInv = np.linalg.inv(Sigma)
	xIn = np.array([X[60], X[160]])
	tIn = np.array([T[60], T[160]])

	for i in range(len(points)):
		xp = xIn[i]
		tp = tIn[i]
		term1 = ((1/sigma2)*np.dot(xp,xp) + SigmaInv)
		mean = (1/sigma2)*np.linalg.inv(term1)*np.dot(xp,tp)
		mean = mean[0]

		term2 = (1/sigma2)*np.dot(xp,xp) + SigmaInv
		Covar = np.linalg.inv(term2)
	

	w0, w1 = np.random.multivariate_normal(mean, Covar, 500000).T
	xmin = w0.min()
	xmax = w0.max()
	ymin = w1.min()
	ymax = w1.max()
	W = np.array((w0, w1)).T
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma', extent = [-2,2,-2,2])
	plt.title('Prior distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	#plt.axis([xmin, xmax, ymin, ymax])
	plt.axis([-2, 2, -2, 2])
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show()

	#print(mean)
	#print(Covar)
	#print(W)
	print(xmin, xmax, ymin, ymax)


def post(X, T):
	p = 60
	xp = X[p]
	tp = T[p]

	alpha = 2   #precision
	Lambda =  np.linalg.inv(np.identity(2)/alpha)
	mu = np.array([0,0])
	A = xp
	L = np.linalg.inv(np.identity(2)*0.2)
	Sigma = np.linalg.inv(Lambda+np.dot(A.T,np.dot(L,A)))
	 
	mean = Sigma*(np.dot(A.T,np.dot(L,tp)) + np.dot(Lambda,mu))
	mean = mean.diagonal()
	Covar = Sigma
	print(mean)
	print()
	print(Sigma)
	w0, w1 = np.random.multivariate_normal(mean, Covar, 500000).T
	xmin = w0.min()
	xmax = w0.max()
	ymin = w1.min()
	ymax = w1.max()
	W = np.array((w0, w1)).T
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma', extent = [-2,2,-2,2])
	plt.title('Prior distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	plt.axis([xmin, xmax, ymin, ymax])
	#plt.axis([-2, 2, -2, 2])
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show()

#post(X,T) 


def test(xIn, tIn):
	n = 160
	x0 = xIn[n]
	x = np.array([x0, 1.0])
	t = tIn[n]
	errorVar = 0.2
	alpha = 2   #precision  
	CovarPrior = np.identity(2)*0.2 #np.identity(2)/alpha   # np.identity(2)*0.2
	CovarPriorInv = np.linalg.inv(CovarPrior)
	precision = CovarPriorInv + (1/errorVar)*np.dot(x,x)
	Covar = np.linalg.inv(precision)

	term1 = (1/errorVar)*np.linalg.inv((1/errorVar)*np.dot(x,x) 
		+ CovarPriorInv)
	term2 = x*t
	mean = np.dot(term1, term2)

	w0, w1 = np.random.multivariate_normal(mean, Covar, 500000).T
	xmin = w0.min()
	xmax = w0.max()
	ymin = w1.min()
	ymax = w1.max()
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma')
	plt.title('Prior distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	#plt.axis([xmin, xmax, ymin, ymax])
	plt.axis([-15, 15, -15, 15])
	hb.set_label('Prior distribution')
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show()
	return(mean)


#s = test(X, T)


def likelihood(xIn, W):
	# variance - small sigma is from the error distribution
	x = np.array([xIn[30], 1.0])
	meanVec = W*x
	var = 0.2
	Covar = np.identity(2)*var
	for i in range(4):
		mean = meanVec[0]
		w0, w1 = np.random.multivariate_normal(mean, Covar, 500000).T
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma')
	plt.title('likelihood distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	xmin = w0.min()
	xmax = w0.max()
	ymin = w1.min()
	ymax = w1.max()
	plt.axis([xmin, xmax, ymin, ymax])
	#plt.axis([-2, 2, -2, 2])
	hb.set_label('Prior distribution')
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show()
	print(W)
	print()
	print(meanVec)
	print(meanVec[0])

#likelihood(X, W)
'''
	w0, w1 = np.random.multivariate_normal(mean, Covar, 50000).T
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma')
	plt.title('Prior distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	#plt.axis([xmin, xmax, ymin, ymax])
	plt.axis([-2, 2, -2, 2])
	hb.set_label('Prior distribution')
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show()
'''






