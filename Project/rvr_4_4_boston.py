# Author: Alice Karnsund and Elin Samuelsson
import numpy as np
from math import pi, exp, sqrt, sin, e
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold


# Gaussian kernel
def RBFMatrix(X1, X2, gamma):
	K = np.ones((len(X1), len(X2) + 1))
	for i in range(len(X1)):
		for j in range(len(X2)):
			K[i,j+1] = exp(-gamma*(np.dot(X1[i]-X2[j],X1[i]-X2[j])**2))
	return (K)


# Prunning called in every iteration of parameter updates, making the training faster
def prune2(gamma, alpha, K, T, Xrv, mu, Trv):
	gamma_toll = 2.22 * 1e-16
	gamma_bool = gamma > gamma_toll
	gamma_bool[0] = True
	gamma = gamma[gamma_bool]

	alpha = alpha[gamma_bool]
	K = K[:, gamma_bool]

	Trv = Trv[gamma_bool[1:]]
	Xrv = Xrv[gamma_bool[1:]]
	mu = mu[gamma_bool]
	#print('Längd gamma:', len(gamma_bool))
	#print('K:', np.shape(K))
	return (gamma, alpha, K, Xrv, mu, Trv)


# Updating relavant parameters and prunes. Returns the optimal parameter values
def updateParams(iterations, K, alpha_start, N, sigma2, T, X):
	alpha_new = alpha_start
	sigma2_new = sigma2
	Trv = np.copy(T)
	Xrv = X
	
	Xrv_list = []
	for i in range(iterations):
		A = np.diag(alpha_new)
		B = (1 / sigma2_new) * np.eye(N)
		term1 = np.dot(np.dot(K.T, B), K) + A
		Sigma = np.linalg.inv(term1)
		gamma = 1 - alpha_new * np.diag(Sigma)
		mu = np.dot(np.dot(np.dot(Sigma, K.T), B), T)
		alpha_new = gamma / (mu ** 2)

		# Prune relevant parameters
		gamma, alpha_new, K, Xrv, mu, Trv = prune2(gamma, alpha_new, K, T, Xrv, mu, Trv)

		# Comment away these 2 when using T1 (no noise) because then sigma2 = konst
		numer = np.dot((T - np.dot(K,mu)),(T - np.dot(K,mu)))
		sigma2_new = numer/(N-sum(gamma))

		# Relevance vectors
		#print('Number of RVs:', len(Xrv), ' at iteration: ', i)	
		if i>=200:
			if Xrv_list[i-200] == len(Xrv):
				break
		Xrv_list.append(len(Xrv))
	#print('Number of RVs:', len(Xrv))
	return (alpha_new, sigma2_new, mu, Sigma, Xrv, Trv)


# Make predictions of unseen x's, x is a vector or matrix
def predictions(X_relevance, mu_s, Sigma_s, sigma2_s, alpha_s, x, gamma_in):
	#K = splineMatrix(x, X_relevance)
	K = RBFMatrix(x, X_relevance, gamma_in)
	mu_s = np.asmatrix(mu_s)
	mean = np.dot(mu_s, K.T)
	#Covar = sigma2_s + np.dot(np.dot(K,Sigma_s),K.T)
	#Covar = np.diag(Covar)
	#t = np.random.normal(mean,Covar)
	#t = t.reshape(-1)
	mean = np.asarray(mean)
	mean = mean.reshape(-1)
	#variance = np.sqrt(Covar)
	#return(t, mean, variance)
	return(mean)


def cross_val(X_in, T_in, kf_in, gamma_in):
	rms_error_list = []
	for train_index, test_index in kf_in.split(X_in):
		X_part, X_part_test = X_in[train_index], X_in[test_index]
		T_part, T_part_test = T_in[train_index], T_in[test_index]
		N = len(T_part)
		iterations = 1000
		sigma2 = 0.01 ** 2
		alpha_start = np.ones(N + 1)  # note a0
		
		K_part = RBFMatrix(X_part, X_part, gamma_in)	
		
		# Train
		alpha_part, sigma2_part, mu_part, Sigma_part, X_part_relevance, T_part_relevance = updateParams(iterations, K_part, alpha_start, N, sigma2, T_part, X_part)
	
		# Predict test data
		mean_part = predictions(X_part_relevance, mu_part, Sigma_part, sigma2_part, alpha_part, X_part_test, gamma_in)

		error_part = mean_part - T_part_test
		rms_error_part = sqrt(sum(error_part**2)/len(error_part))
		rms_error_list.append(rms_error_part)
	score = sum(rms_error_list)
	print(score)
	return score	
	

def generate_initial(X_in, T_in, kf_in):
	best_gamma = 1.0	
	# Friedman 1	
	#gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
	# Friedman 2
	#gammas = [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10]
	# Friedman 3
	gammas = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
	min_score = 1e10
	for g in gammas:
		score = cross_val(X_in, T_in, kf_in, g)
		if score < min_score:
			best_gamma = g
			min_score = score
	print('min score: ', min_score, 'for gamma: ', best_gamma)
	return(best_gamma, min_score)
			
def find_opt_gamma(X_in, T_in, kf_in, best_gamma, min_score):
	gammas = [0.25*best_gamma, 0.5*best_gamma, 0.75*best_gamma, 2.5*best_gamma, 5.0*best_gamma, 7.5*best_gamma]
	for g in gammas:
		score = cross_val(X_in, T_in, kf_in, g)
		if score < min_score:
			best_gamma = g
			min_score = score
	print('min score: ', min_score, 'for gamma: ', best_gamma)
	return(best_gamma, min_score)




# ---------------------------------- RUN section ---------------------------------

# Settings
N = 481
iterations = 1000
sigma2 = 0.01 ** 2
alpha_start = np.ones(N + 1)  # note a0

# Test data 
total_X, total_T = datasets.load_boston(n_samples=1000)

gamma_list = []
rms_error_list = []
sup_vec_list = []
for iter in range(20):
	print('data set: ', iter+1)

	# Training data
	X, T, test_X, test_T = train_test_split(total_X, total_T, test_size=481)

	kf = KFold(n_splits=5, shuffle=True)
	
	init_gamma, init_score = generate_initial(X, T, kf)
	print('initial: ', init_gamma, ' with score: ', init_score)
	
	opt_gamma, opt_score = find_opt_gamma(X, T, kf, init_gamma, init_score)
	print('optimal: ', opt_gamma, ' with score: ', opt_score)
	gamma_list.append(opt_gamma)

	K = RBFMatrix(X,X, opt_gamma)	
	# Get optimal parameters and relevance vectors
	alpha_s, sigma2_s, mu_s, Sigma_s, X_relevance, T_relevance = updateParams(iterations, K, alpha_start, N, sigma2, T, X)
	# Predict test data
	mean = predictions(X_relevance, mu_s, Sigma_s, sigma2_s, alpha_s, test_X, opt_gamma)

	error = mean - test_T
	max_error = max(error)
	rms_error = sqrt(sum(error**2)/len(error))
	rms_error_list.append(rms_error)

	count_sup_vec = len(X_relevance)
	sup_vec_list.append(count_sup_vec)

print('RMS: ', rms_error_list)
print('sup vec: ', sup_vec_list)
print('gamma: ', gamma_list)

av_rms_error = sum(rms_error_list)/len(rms_error_list)
print('RMS: ', rms_error)
av_sup_vec = sum(sup_vec_list)/len(sup_vec_list)
print('sup vec: ', av_sup_vec)
av_gamma = sum(gamma_list)/len(gamma_list)
print('sup vec: ', av_gamma)
