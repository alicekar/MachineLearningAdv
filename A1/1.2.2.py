import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi, e, exp
from scipy.spatial.distance import cdist

def data():
	N = 100
	X = np.linspace(-4,4,N)
	return(X)

def kernel(X,l=1):
	sigma = 1
	K = np.zeros((len(X),len(X)))
	for i in range(len(X)):
		for j in range(len(X)):
			xi = X[i]
			xj =X[j]
			ker = (sigma**2)*exp(-np.dot((xi-xj),(xi-xj))/(l**2)) 
			K[i,j] = ker
			j+=1
		i+=1
	return(K)
X = data()


def prior(l,X):
    mean = np.zeros(len(X)) # vector of the means
    K = kernel(X,l)
    samples = 10
    f = np.random.multivariate_normal(mean,K,samples)
    for i in range(samples):
        pb.plot(X[:],f[i,:])
    plt.axis([-4, 4, -4, 4])
    plt.title('GP-prior with length scale '+str(l))
    plt.show()

prior(100,X)




