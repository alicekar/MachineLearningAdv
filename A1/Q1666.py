import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from math import pi, sin, cos, e
import scipy.optimize as opt



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

	Y = np.dot(f_nonlin, A.T)    # Y = f_lin(f_nonlin(x)), f_lin(x') = x'A^T
	rankY = np.linalg.matrix_rank(Y)
	return(Y,x)




def f(x, *args):
	# return the value of the objective at x
	# Linear relation: Y = XW^T, optimize W
	W = x
	W = np.squeeze(np.asmatrix(W)).T
	N = Y.shape[0]
	D = Y.shape[1]
	term1 = ((N*D)/2)*np.log(2*pi)
	term2 = (N/2)*np.log(np.linalg.det(np.dot(W,W.T) + np.eye(D)))
	inner = np.linalg.inv(np.dot(W,W.T) + np.eye(D))
	term3 = (1/2)*np.trace(np.dot(inner,np.dot(Y.T,Y)))
	val = term1 + term2 + term3
	return val




def dfx(x,*args):
	# return the gradient of the objective at x
	W = x
	W = np.squeeze(np.asmatrix(W)).T
	N = Y.shape[0]
	D = Y.shape[1]
	gradient = np.zeros([D,1])

	for i in range(D):
		wi = W[i]
		# create single matrix
		single = np.zeros([D,1])
		single[i] = 1
		term11 = np.linalg.inv(np.dot(W,W.T)+np.eye(D))
		term12 = np.dot(single,W.T)+np.dot(W,single.T)
		term1 = (N/2)*np.trace(np.dot(term11,term12))

		term21 = term11
		term22 = np.dot(single,W.T)+np.dot(W,single.T)
		term23 = term21
		termY = np.dot(Y.T,Y)
		term2 = (-1/2)*np.trace(np.dot(termY,np.dot(term21,np.dot(term22,term23))))

		grad_Elem = term1 + term2 
		gradient[i] = grad_Elem
		i+=1 
	val = np.squeeze(np.asarray(gradient))
	v = np.squeeze(np.asmatrix(val))
	print(val.shape)
	return val


Y, x = generateData()
print(Y.shape)
W0 = np.linspace(0,9,10)
W0 = np.array([W0]).T

W0 = np.asarray((1,1,1,1,1,1,1,1,1,1))
print(W0.shape)
x0 = W0
args = Y
dfx(W0,Y)
'''
x_star = opt.fmin_cg(f,x0,fprime=dfx, args=args)
W_star = x_star
W_star = np.squeeze(np.asmatrix(W_star)).T

inner = np.linalg.inv(np.dot(W_star.T, W_star))
x_approx = np.dot(Y, np.dot(W_star,inner))
#print(W_star)
#print(x)
xa = np.squeeze(np.asarray(x_approx))
#print(xa)

n = np.linspace(0,99,100)
plt.plot(x[:], x[:], 'b')
#plt.show()
plt.plot(x[:], xa[:], 'r')
plt.xlabel('n = dimension')
plt.ylabel('blue = true x, red = approx x')
#plt.show()
'''


'''
X = np.linspace(0,4*pi, 100)
X = np.array([X]).T
print(W.T.shape)
print(np.dot(X, W.T).shape)
term = np.linalg.inv(np.dot(W.T, W))
term1 = (np.dot(W, term))
x = np.dot(Y,term1)
print(x.shape)'''

