# Author: Alice Karnsund and Elin Samuelsson
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, e, sqrt
import random as rd
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def sinc(x):
	abs_x = abs(x)	
	if abs_x == 0:
		result = 1
	else:
		result = sin(abs_x)/abs_x
	return result

def sinc_gaussian(x, st_dev):
	abs_x = abs(x)	
	if abs_x == 0:
		result = 1 + np.random.normal(scale=st_dev)
	else:
		result = sin(abs_x)/abs_x + np.random.normal(scale=st_dev)
	return result

def sinc_uniform(x, width):
	abs_x = abs(x)	
	if abs_x == 0:
		result = 1 + np.random.uniform(low=-width, high=width)
	else:
		result = sin(abs_x)/abs_x + np.random.uniform(low=-width, high=width)
	return result

def generate_sinc_data(in_data):
	out_data = []	
	for i in in_data:
		out_data.append(sinc(i))
	return out_data

def generate_sinc_gaussian_data(in_data, st_dev):
	out_data = []	
	for i in in_data:
		out_data.append(sinc_gaussian(i, st_dev))
	return out_data

def generate_sinc_uniform_data(in_data, width):
	out_data = []	
	for i in in_data:
		out_data.append(sinc_uniform(i, width))
	return out_data

def linear_spline_kernel(X1,X2):
	N1 = X1.shape[0]
	N2 = X2.shape[0]
	dim = X1.shape[1]

	X1_abs = np.absolute(X1)
	X2_abs = np.absolute(X2)
	
	result = np.ones((N1, N2))
	for i in range(N1): 
		for j in range(N2):
			for d in range(dim):
				result[i,j] *= 1 + X1_abs[i,d]*X2_abs[j,d] + min(X1_abs[i,d],X2_abs[j,d])**2*abs(X1_abs[i,d]-X2_abs[j,d])/2 + min(X1_abs[i,d],X2_abs[j,d])**3/3
				#result[i,j] *= 1 + X1[i,d]*X2[j,d] + min(X1[i,d],X2[j,d])**2*abs(X1[i,d]-X2[j,d])/2 + min(X1[i,d],X2[j,d])**3/3
	return result

#__________________________________________________________________________________________________________________
# EX. 4.4

test_input = np.arange(-10,10,0.02).reshape(-1, 1)
test_output = np.array(generate_sinc_data(test_input)).reshape(1000)

sinc_input = np.arange(-10,10,0.2).reshape(-1, 1)

#_________________________________________________________________________________________________________________
# UNIFORM NOISE 0.1

uniform_rms_error = []
uniform_sup_vec = []

for iter in range(20):
	print(iter)
	sinc_output = np.array(generate_sinc_gaussian_data(sinc_input, 0.1)).reshape(100)

	print(max(sinc_output))
	print(min(sinc_output))

	parameters = {'C':[1e-2, 1e0, 1e2], 'epsilon':[1e-4, 1e-3, 1e-2, 1e-1], 'gamma':[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]}

	svr = SVR(kernel='rbf')
	clf = GridSearchCV(svr, parameters, cv=5)
	clf.fit(sinc_input, sinc_output) 
	print("Best parameters: ", clf.best_params_)

	interval1 = np.array([0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
	interval2 = np.array([0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5])
	c_best = clf.best_params_['C']
	epsilon_best = clf.best_params_['epsilon']
	gamma_best = clf.best_params_['gamma']	

	c_params = c_best*interval1
	epsilon_params = epsilon_best*interval2
	gamma_params = gamma_best*interval2

	parameters = {'gamma':gamma_params}
	svr = SVR(kernel='rbf', C=c_best, epsilon=epsilon_best)
	clf = GridSearchCV(svr, parameters, cv=5)
	clf.fit(sinc_input, sinc_output) 
	print("Best parameters: ", clf.best_params_)
	gamma_best = clf.best_params_['gamma']	

	parameters = {'C':c_params}
	svr = SVR(kernel='rbf', epsilon=epsilon_best, gamma=gamma_best)
	clf = GridSearchCV(svr, parameters, cv=5)
	clf.fit(sinc_input, sinc_output) 
	print("Best parameters: ", clf.best_params_)
	c_best = clf.best_params_['C']

	parameters = {'epsilon':epsilon_params}
	svr = SVR(kernel='rbf', C=c_best, gamma=gamma_best)
	clf = GridSearchCV(svr, parameters, cv=5)
	clf.fit(sinc_input, sinc_output)  
	print("Best parameters: ", clf.best_params_)
	epsilon_best = clf.best_params_['epsilon']


	test_pred = clf.predict(test_input)

	error = test_pred - test_output
	max_error = max(error)
	rms_error = sqrt(sum(error**2)/len(error))
	uniform_rms_error.append(rms_error)

	count_sup_vec = len(clf.best_estimator_.support_)
	uniform_sup_vec.append(count_sup_vec)

print('RMS: ', uniform_rms_error)
print('sup vec: ', uniform_sup_vec)

average_uniform_rms_error = sum(uniform_rms_error)/len(uniform_rms_error)
print('RMS: ', average_uniform_rms_error)
average_uniform_sup_vec = sum(uniform_sup_vec)/len(uniform_sup_vec)
print('sup vec: ', average_uniform_sup_vec)

















