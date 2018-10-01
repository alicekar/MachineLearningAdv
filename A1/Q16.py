import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, sin, cos, e
import scipy.optimize as opt

# Task: recover x' – 100x2, given Y 
# optimize W (aka A) 2x10 

def generateData():
	# create test data
	A = np.zeros([10,2])
	x = np.linspace(0,4*pi,100)
	f_nonlin = np.zeros([100, 2])
	
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			Aij = np.random.normal(0,1)
			A[i][j] = Aij
			j += 1
		i += 1
	
	for i in range(f_nonlin.shape[0]):
		fi1 = x[i]*sin(x[i])
		fi2 = x[i]*cos(x[i])
		f_nonlin[i][0] = fi1
		f_nonlin[i][1] = fi2
		i += 1

	X_prim = f_nonlin
	Y = np.dot(f_nonlin, A.T)    # Y = f_lin(f_nonlin(x)), f_lin(x') = x'A^T
	return(Y, A, X_prim)
	



def f(x, *args):
	# return the value of the objective at x
	# assuming sigma = 1, and D is 1
	W = np.array([x[0:10],x[10:20]]).T
	N = Y.shape[0]      # 100
	D = Y.shape[1]		# 10		

	term1 = ((N*D)/2)*np.log(2*pi)
	term2 = (N/2)*np.log(np.linalg.det(np.dot(W, W.T) + np.eye(D)))
	inner = np.linalg.inv(np.dot(W, W.T) + np.eye(D))
	term3 = (1/2)*np.trace(np.dot(inner, np.dot(Y.T,Y)))   #np.dot(Y,Y.T)??
	val = term1 + term2 + term3
	print(val)
	return val




def dfx(x, *args):
	# return the gradient of the objective at x
	W = np.array([x[0:10],x[10:20]]).T
	N = Y.shape[0]      # 100
	D = Y.shape[1]	
	gradient = np.zeros(W.shape)
	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			single = np.zeros(W.shape)    # single matrix, zero everywhere except i,j = 1
			single[i][j] = 1

			term11 = np.linalg.inv(np.dot(W, W.T) + np.eye(D))
			term12 = (np.dot(single, W.T) + np.dot(W, single.T))
			term1 = (N/2)*np.trace(np.dot(term11, term12))
			term21 = term11
			term22 = np.dot(single, W.T) + np.dot(W, single.T)
			term23 = term21
			termY = np.dot(Y.T, Y)
			term2 = (1/2)*np.trace(np.dot(termY, np.dot(term21, np.dot(term22, term23))))
			derivative = term1 - term2
			gradient[i][j] = derivative
			j += 1
		i += 1
	val = gradient.flatten()
	print(val.shape)
	return val


Y, W , X_prim = generateData()
W0 = np.ones([10,2])
W0 = W0.flatten().T
W0 = np.random.randn(20)
W0 = W0.reshape((10, 2)).flatten().T
args = Y


W_star = opt.fmin_cg(f,W0,fprime=dfx, args=(Y,))   # x_star = A_star

W_star = W_star.reshape((2,10)).T

inner = np.linalg.inv(np.dot(W_star.T, W_star))
X_approx = np.dot(Y, np.dot(W_star,inner))
print(X_approx[50:60])
print(X_prim[50:60])
line1, = plt.plot(X_approx[:,0], X_approx[:,1], label="Learned $X'$", color ='b')
line2, = plt.plot(X_prim[:,0], X_prim[:,1], label="True $X'$", color = 'r')
fl = plt.legend(handles=[line1], loc=1)
ax = plt.gca().add_artist(fl)
plt.legend(handles=[line2], loc=4)
plt.xlabel("$X'$[:,0]")
plt.ylabel("$X'$[:,1]")
plt.title("Learned and true $X'$-values ")
plt.show()















