import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi, e, exp, cos
from scipy.spatial.distance import cdist

def data():
	N = 7       #number of given data points
	N1 = 500	# number of points for showing the true function
	X1 = np.linspace(-10*pi,10*pi,N1)    # points for true function
	X = np.linspace(0,2*pi,N)    # given data points
	T = np.zeros(N)     # here the corresponing target values will be
	Y = np.zeros(N)		# here the 7 true function values will be
	Y1 = np.zeros(N1)	# here the true function values will be
	for j in range(N1):
		yj = cos(X1[j])
		Y1[j] = yj
		j+=1
	for i in range(N):
		noise = np.random.normal(0, 0.5)
		ti = cos(X[i]) + noise
		yi = cos(X[i])
		T[i] = ti
		Y[i] = yi
		i+=1
	pb.plot(X1[:], Y1[:], 'blue')	  # plot true function
	#pb.plot(X[:], Y[:], 'bo')	# plot true values for the 7 x's
	pb.plot(X[:], T[:], 'go')	# plot target values for the 7 x's
	#pb.show()

	return(X,T)

X, T = data()


def kernel1(X,Y,l=1):
	sigma = 1
	K = np.zeros((len(X),len(Y)))
	for i in range(len(X)):
		for j in range(len(Y)):
			xi = X[i]
			xj =Y[j]
			ker = (sigma**2)*exp(-np.dot((xi-xj),(xi-xj))/(l**2)) 
			K[i,j] = ker
			j+=1
		i+=1
	return(K)


def prior(l,X):
    mean = np.zeros(len(X)) # vector of the means
    xmin = X.min()
    xmax = X.max()
    X = X[:,None]
    K = kernel1(X, X, l)
    samples = 10
    f = np.random.multivariate_normal(mean,K,samples)
    # every row coressponds to the values of a specific f
    print(f)
    for i in range(samples):
        pb.plot(X[:],f[i,:])
    pb.axis([xmin, xmax, -4, 4])
    pb.title('GP-prior with length scale '+str(l))
    pb.show()
#prior(1,X)


def computePosterior(X,T,l):
    #Posterior predictice distribution - distribution of possible unobserved values conditional on the observed values
    N = 800
    x = np.linspace(-10*pi,10*pi,N)
    x_star = x[:, None]
    X = X[:,None]
    k = kernel1(x_star,X,l)
    variance = 0.5
    N = len(X)
    C = np.linalg.inv(kernel1(X,X,l)+ np.eye(N,N)*variance)
    t = T[:,None]
    mu = np.dot(np.dot(k,C),t)
    c = kernel1(x_star, x_star, l)
    sigma = c - np.dot(np.dot(k,C),np.transpose(k))
    
    # plot mu and sigma
    Mu = np.zeros(len(mu))
    for i in range(len(mu)):
    	Mu[i] = mu[i]
    	i += 1	

    S = sigma.diagonal()
    std = np.sqrt(S)
    std = std[:,None]
    STD = np.zeros(len(std))
    for i in range(len(mu)):
    	STD[i] = std[i]
    	i += 1	
    stdUpper = Mu + STD
    stdLower = Mu - STD
    stdLower = np.array(stdLower)
    stdUpper = np.array(stdUpper)
    pb.fill_between(x, stdLower, stdUpper, color= 'grey', alpha = '0.5')
    pb.plot(x_star[:],Mu[:], 'red')
    pb.axis([-6, 10, -3, 3])
    pb.title('Data, predictive mean and variance of the posterior')
    pb.show()
    return Mu, sigma, x

mu, sigma, points = computePosterior(X,T,1)


def post(l,points, mu, sigma):
    xmin = points.min()
    xmax = points.max()
    points = points[:,None]
    K = kernel1(points, points, l)
    samples = 3
    colors = ('orange','red','pink')
    f = np.random.multivariate_normal(mu,sigma,samples)
    # every row coressponds to the values of a specific f
    for i in range(samples):
        pb.plot(points[:],f[i,:],colors[i])
    #pb.axis([-6, 10, -3, 3])
    pb.title('GP-posterior with length scale '+str(l))
    #pb.show()
#post(1,points, mu, sigma)

#plotPosterior(mu, sigma, points)



'''

def posterior(l,X,T):
	N = len(X)
	X = X[:,None]
	points = np.linspace(0,2*pi,100)
	variance = 0.5
	CN = kernel1(X,X,l) + np.eye(N,N)*variance
	CNinv = np.linalg.inv(CN)
	#F = np.zeros(len(points))
	Mean = np.zeros(len(points))

	for i in range(len(points)):
		p = np.array([points[i]])
		p = p[:,None]
		k = kernel1(X,p,l)
		mean = np.dot(k.T, np.dot(CNinv,T))
		c = kernel1(p,p) + variance
		covar = c- np.dot(k.T, np.dot(CNinv,k))
		#f = np.random.multivariate_normal(mean,covar)
		#F[i] = f
		M[i] = mean
		i += 1
		
	print(F)
	pb.plot(points[:],F[:])
	pb.plot(points[:], M[:])
	pb.axis([0,6.5,-4,4])
	pb.show()
#posterior(1,X,T)

'''

