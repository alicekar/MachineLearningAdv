# Author: Alice Karnsund
import numpy as np
from math import pi, exp, sqrt
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel 


# Generate som test data for the 2 pictures
def generateData(N):
	X = np.linspace(-10,10,N)
	# without noise
	T1 = (1/np.absolute(X))*np.sin(np.absolute(X))
	# with noise
	T2 = np.copy(T1)
	for t in range(len(T2)):
		error = np.random.normal(0, 0.2)
		T2[t] = T2[t]+error	
	return(X, T1, T2)


# Linear spline kernel
def splineMatrix(Y,X):
	N = len(X)
	matrix = np.ones((len(Y),N+1))
	# K_mn = K(x_m,x_n-1)    obs note m=n and n=m in paper
	for m in range(len(Y)):
		x_m = Y[m]
		for n in range(N):
			x_n = X[n]
			elem = 1 + x_m*x_n + x_m*x_n*min(x_m,x_n) - 0.5*(x_m+x_n)*min(x_m,x_n)**2
			elem = elem + (min(x_m,x_n)**3)/3
			matrix[m][n+1] = elem
	return(matrix)


# Gaussian kernel
def RBFMatrix(Y,X):
	K = np.ones((len(Y),len(X)+1))
	X = np.asmatrix(X).T
	Y = np.asmatrix(Y).T
	k = rbf_kernel(Y,X)
	K[:,1:] = k
	print(np.shape(k))
	print(np.shape(K))
	return(K)


# Prunning called in every iteration of parameter updates, making the training faster
def prune(gamma, alpha, K, T, X, mu):
	gamma_toll = 2.22*1e-15
	gamma_bool = gamma > gamma_toll
	gamma_bool[0] = True
	gamma = gamma[gamma_bool]
	N_prune = len(gamma)-1
	alpha = alpha[gamma_bool]
	K = K[:, gamma_bool] 
	K = K[gamma_bool[1:],:]
	T = T[gamma_bool[1:]]
	RVs = X[gamma_bool[1:]]
	mu = mu[gamma_bool]
	print('längd gamma:',len(gamma_bool))
	print('K:', np.shape(K))
	return(gamma, alpha, K, T, mu, N_prune, RVs)
	

# Updating relavant parameters and prunes. Returns the optimal parameter values
def updateParams(K, alpha_start, N, sigma2, T, X):
	# Number of relevance vectors clearly depend on number of iterations
	# First Picture: 800 iterations for T1 and sigma2=0.01^2 to give 9 RVs, remember
	# to comment away line 90 and 91 ;)
	iterations = 300

	alpha_new = alpha_start
	sigma2_new = sigma2

	for i in range(iterations):
		A = np.diag(alpha_new)
		B = (1/sigma2_new)*np.eye(N)
		term1 = np.dot(np.dot(K.T,B),K)+A
		Sigma = np.linalg.inv(term1)
		gamma = np.ones(N+1)-alpha_new*np.diag(Sigma)
		mu = np.dot(np.dot(np.dot(Sigma,K.T),B),T)
		alpha_new = gamma/(mu**2)

		# Prune relevant parameters 
		gamma, alpha_new, K, T, mu, N, RVs  = prune(gamma, alpha_new, K, T, X, mu)

		# Comment away these 2 when using T1 (no noise) because then sigma2 = konst
		numer = np.dot((T - np.dot(K,mu)),(T - np.dot(K,mu)))
		sigma2_new = numer/(N-sum(gamma))

		# Relevance vectors
		X = RVs
		print('Number of RVs:', len(RVs))
	return(alpha_new, sigma2_new, mu, Sigma, X, T, gamma)


# Make predictions of unseen x's, x is a vector or matrix
def predictions(X_relevance, mu_s, Sigma_s, sigma2_s, alpha_s, x):
    K = splineMatrix(x, X_relevance)
    
    mu_s = np.asmatrix(mu_s)
    print(np.shape(mu_s))
    print(np.shape(K))
    mean = np.dot(mu_s, K.T)
  
    Covar = sigma2_s + np.dot(np.dot(K,Sigma_s),K.T)
    Covar = np.diag(Covar)
    t = np.random.normal(mean,Covar)
    return(t)



# Number of data points
N = 100
sigma2 = 0.01**2 
alpha_start = np.ones(N+1)  # note a0
X, T1, T2 = generateData(N)
K = splineMatrix(X,X)
#K = RBFMatrix(X,X)

# Change depending on data set
T = T2

# Get optimal parameters and relevance vectors
alpha_s, sigma2_s, mu_s, Sigma_s, X_relevance, T_relevance, gamma = updateParams(K, alpha_start, N, sigma2, T, X)
print('mu:', mu_s)
print('gamma: ', gamma)
print('alpha: ', alpha_s)
print('Sigma: ', Sigma_s)
print('sigma2:', sigma2_s)
# Correct function
plt.plot(X, T1)    

# Plot data points, change depending on data set
plt.plot(X, T2, 'ro', markersize = 2)
#plt.plot(X, T1, 'ro', markersize = 2)

# Plot relevance vectors
plt.plot(X_relevance, T_relevance, 'ro')

'''
# Plot predictions
test = np.linspace(-10,10,50)
for i in range(len(test)):
	x = test[i]
	t = makePredictions(X_relevance, mu_s, Sigma_s, sigma2_s, alpha_s, x)
	plt.plot(test[i], t, 'gx')
	print(t)

'''
test = np.linspace(-10,10,50)
l = predictions(X_relevance, mu_s, Sigma_s, sigma2_s, alpha_s, test )
test = np.asmatrix(test)
print(np.shape(test))
print(np.shape(np.asarray(l)))
plt.plot(test, l, 'gx')

plt.show()


