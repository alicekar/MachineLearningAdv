# Author: Alice Karnsund
import numpy as np
from math import pi, exp, sqrt
import matplotlib.pyplot as plt
from scipy.special import gamma 

# Our goal is to infer the posterior distribution for the 
# mean mu and precision tau

# generate some data drawn independently from a Gaussian 
def generateData(N):
	# Goal values
	mu = 0
	sigma = 1    # tau = 1/sigma = 1 
	D = np.random.normal(mu, sigma, N)
	return(D)


# Compute the moments
def moments(aN, bN, muN, lambdaN):
	E_mu = muN      # constant
	E_mu2 = (1/lambdaN) + muN**2
	E_tau = aN/bN
	return(E_mu, E_mu2, E_tau)


# Posterior approximation
def q(aN, bN, muN, lambdaN, mu, tau):
	q_mu = (1/gamma(aN))*(bN**aN * tau**(aN-1) * np.exp(-bN*tau)) 
	q_tau = (lambdaN/(2*pi))**(1/2) 
	q_tau = q_tau * np.exp(-0.5*np.dot(lambdaN,((mu-muN)**2).T)) 
	q = q_mu*q_tau
	return(q)


# plot the contours for a distribution
def plotPost(mu, tau, approx, color, i):
	muGrid, tauGrid = np.meshgrid(mu, tau)
	plt.contour(muGrid, tauGrid, approx, colors = color)
	plt.title('Posterior approximation after '+str(i)+' iterations')
	plt.axis([-1.,1.,0,2])  # not always necessary 
	plt.xlabel('$\mu$')
	plt.ylabel('tau')
	plt.show()


# Compute the true posterior parameters
def trueParameters(N, D, a0, b0, mu0, lambda0):
	x_mean = (1/N)*sum(D)
	muT = (lambda0*mu0 + N*x_mean)/(lambda0 + N)
	lambdaT = lambda0 + N
	aT = a0 + N/2
	bT = b0 + 0.5*sum((D-x_mean)**2)
	bT = bT + (lambda0*N*(x_mean-mu0)**2)/(2*(lambda0+N))
	return(aT, bT, muT, lambdaT)


# Compute the true posterior distribution
def truePost(aT, bT, muT, lambdaT, mu, tau):
	post = ((bT**aT)*sqrt(lambdaT))/(gamma(aT)*sqrt(2*pi)) 
	post = post*(tau**(aT-0.5))*np.exp(-bT*tau)
	post = post*np.exp(-0.5*(lambdaT*np.dot(tau,((mu-muT)**2).T)))
	return(post)


# Plot the true posterior distribution
def plotTrue(mu, tau, true):
	muGrid, tauGrid = np.meshgrid(mu, tau)
	plt.contour(muGrid, tauGrid, true)


# VI step, update parameters 
def updateParameters(N, D, a0, b0, mu0, lambda0, bN, lambdaN, mu, tau):
	# Some parameter values will be constant
	x_mean = (1/N)*sum(D)
	muN = (lambda0*mu0 + N*x_mean)/(lambda0 + N)
	aN = a0 + (N+1)/2
	iterations = 6    

	# True posterior parameters
	aT, bT, muT, lambdaT = trueParameters(N,D,a0,b0,mu0,lambda0)

	for i in range(iterations):
		E_mu, E_mu2, E_tau = moments(aN, bN, muN, lambdaN)
		lambdaN = (lambda0+N)*E_tau
		bN = b0-E_mu*(lambda0*mu0+sum(D))
		bN = bN+0.5*(E_mu2*(lambda0+N)+lambda0*mu0**2+sum(D**2))

		# Calculate q for new values of the parameters
		approx = q(aN,bN,muN,lambdaN,mu[:,None],tau[:,None])

		# Calculate the true posterior 
		post = truePost(aT,bT,muT,lambdaT,mu[:,None],tau[:,None])

		# Plot the posterior approximation and the true 
		plotTrue(mu, tau, post)
		color = 'b'
		if i == iterations-1:
			color = 'r'

		plotPost(mu, tau, approx, color, i)



# Generate some data points
N = 10
D = generateData(N)

# Choose parameter values for the conjugate prior
a0 = 3 #0
b0 = 5 #0
mu0 = 0.2 #0
lambda0 = 15 #0

# Start values
bN = 3 #0.1 
lambdaN = 15 #0.2 

# Generate some mu and tau values for the plot 
mu = np.linspace(-2,2,100)
tau = np.linspace(0,4,100)

# Run VI
updateParameters(N, D, a0, b0, mu0, lambda0, bN, lambdaN, mu, tau)











