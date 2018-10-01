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
	error = np.random.normal(0,errorVar)
	T = np.zeros(N)
	for i in range(N):
		t = Wtrue[0]*X[i] + Wtrue[1] + error
		#plt.plot(X[i],t,'bo')
		T[i] = t
		i += 1
	#plt.show()
	return(X,T)

#print(generateData())
'''
plt.axis([-2,2,-3,3])
plt.figure(1)

for i in range(4):	
	error = np.random.normal(0,0.2)
	t = Wtrue[0]*X + Wtrue[1] + error
	colors = ['ro', 'bo', 'go', 'yo']
	plt.plot([X[1],X[100],X[200]],[t[1],t[100],t[200]], colors[i])
	plt.plot(X, t, colors[i])
	i += 1
plt.show()

'''
def prior(): 
	mean = np.array([0,0])
	alpha = 2   #precision  
	Covar = np.identity(2)/alpha   # np.identity(2)*0.2

	w0, w1 = np.random.multivariate_normal(mean, Covar, 5000000).T
	W = np.array((w0, w1)).T
	xmin = w0.min()
	xmax = w0.max()
	ymin = w1.min()
	ymax = w1.max()
	return(W)
'''
	hb = plt.hexbin(w0, w1, gridsize=100, cmap='plasma')
	plt.title('Prior distribution')
	plt.xlabel('$w_0$')
	plt.ylabel('$w_1$')
	#plt.axis([xmin, xmax, ymin, ymax])
	plt.axis([-2, 2, -2, 2])
	cb = plt.colorbar(hb)
	cb.set_label('counts')
	plt.show()
'''
	

W = prior()


def test(xIn, tIn):
	n = 160
	x0 = xIn[n]
	x = np.array([x0, 1.0])
	t = tIn[n]
	errorVar = 0.2
	alpha = 2   #precision  
	CovarPrior = np.identity(2)/alpha   # np.identity(2)*0.2
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

#X,T = generateData()
#s = test(X, T)


def likelihood(xIn, W):
	# variance - small sigma is from the error distribution
	x = np.array([xIn[30], 1.0])
	meanVec = W*x
	var = 0.2
	Covar = np.identity(2)*var
	for i in range(4):
		mean = meanVec[0]
		w0, w1 = np.random.multivariate_normal(mean, Covar, 50000).T
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






