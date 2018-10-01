import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 	
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# To sample from a multivariate Gaussian
#f = np.random.multivariate_normal(mu,K)

# To compute a distance matrix between two sets of vectors
#D = cdist(x1,x2)

# To compute the exponetial of all elements in a matrix
#E = np.exp(D)

def plot_2d_gaussian(mu, Sigma):
	# Our 2-dimensional distribution will be over variables X and Y
	N = 60
	X = np.linspace(-3, 3, N)
	Y = np.linspace(-3, 3, N)
	X, Y = np.meshgrid(X, Y)

	# Pack X and Y into a single 3-dimensional array
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X
	pos[:, :, 1] = Y

	def multivariate_gaussian(pos, mu, Sigma):

	    n = mu.shape[0]
	    Sigma_det = np.linalg.det(Sigma)
	    Sigma_inv = np.linalg.inv(Sigma)
	    N = np.sqrt((2*np.pi)**n * Sigma_det)
	    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	    # way across all the input variables.
	    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

	    return np.exp(-fac / 2) / N

	# The distribution on the variables X, Y packed into pos.
	Z = multivariate_gaussian(pos, mu, Sigma)
	print(mu)
	print(Sigma)
	# Create a surface plot and projected filled contour plot under it.
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
		        cmap=cm.viridis)

	cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.5, cmap=cm.viridis)

	# Adjust the limits, ticks and view angle
	ax.set_zlim(-0.5,1.0)
	ax.set_zticks(np.linspace(0,1.0,5))
	ax.view_init(27, -21)

	plt.title('Posterior distribution over weights, given ten data points')
	plt.xlabel('w0')
	plt.ylabel('w1')
	plt.show()

#___________________________________________________________________________________

w0 = 1.5
w1 = -0.8
mean = [0]
cov = [0.2]

x = np.arange(-2,2,0.02)
epsilon = np.random.normal(mean,cov, 200)
t = w0*x + w1*np.ones(200) + epsilon
data = [x,t]
print(t)
#plt.plot(x,t,'r.')
#plt.xlabel('input x')
#plt.ylabel('output t')
#plt.title('Data points (x,t)')
#plt.show()

# Prior over weights
mu1 = np.array([0., 0.])
Sigma1 = np.array([[ 1. , 0.], [0.,  1.]])

weights = np.array([np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1), np.random.multivariate_normal(mu1, Sigma1)])

#print(weights)
for i in range(10):
	t_approx = weights[i][0]*x + weights[i][1]*np.ones(200)
	#plt.plot(x, t_approx)
#plt.title('Functions drawn from the prior')
#plt.xlabel('input x')
#plt.ylabel('output t')

#plot_2d_gaussian(mu1, Sigma1)





# one sample:
sample = [x[1], t[1]]
cov_sample = np.identity(2) + np.array([[sample[0]*sample[0], sample[0]], [sample[0], 1]])
cov_sample = np.linalg.inv(cov_sample)
mean_sample = sample[0]*np.dot(cov_sample, np.array([[sample[0]],[1]])).reshape(1,2)
print('sample', sample)
print('cov', cov_sample)
print('mean', mean_sample)
#plot_2d_gaussian(mean_sample, cov_sample)

mean_sample = mean_sample.reshape(2)
weights = np.array([np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample)])

# print(weights)
for i in range(10):
	t_approx = weights[i][0]*x + weights[i][1]*np.ones(200)
	plt.plot(x, t_approx)
plt.plot(sample[0], sample[1], 'ro')

plt.xlabel('input x')
plt.ylabel('output t')
plt.title('Functions drawn from the posterior, given one data point')
plt.show()






# two samples: 
x_mat = np.array([[x[10],1],[x[150],1]])
print(x_mat)
t_mat = np.array([[t[10]],[t[150]]])
print(t_mat)
cov_sample = np.identity(2) + np.dot(np.transpose(x_mat),x_mat)
cov_sample = np.linalg.inv(cov_sample)
mean_sample =np.dot(cov_sample, np.dot(np.transpose(x_mat),t_mat)).reshape(1,2)
#print(cov_sample)
#print(mean_sample)
#plot_2d_gaussian(mean_sample, cov_sample)

mean_sample = mean_sample.reshape(2)
weights = np.array([np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample)])

# print(weights)
for i in range(10):
	t_approx = weights[i][0]*x + weights[i][1]*np.ones(200)
	#plt.plot(x, t_approx)
for i in range(2):
	#plt.plot(x_mat[i][0], t_mat[i][0], 'ro')
	print()

#plt.xlabel('input x')
#plt.ylabel('output t')
#plt.title('Functions drawn from the posterior, given two data points')
#plt.show()
'''





# ten samples: 
x_mat = np.array([[x[10],1],[x[30],1],[x[45],1],[x[70],1],[x[80],1],[x[105],1],[x[110],1],[x[150],1],[x[160],1],[x[195],1]])
t_mat = np.array([[t[10]],[t[30]],[t[45]],[t[70]],[t[80]],[t[105]],[t[110]],[t[150]],[t[160]],[t[195]]])

cov_sample = np.identity(2) + np.dot(np.transpose(x_mat),x_mat)
cov_sample = np.linalg.inv(cov_sample)
mean_sample =np.dot(cov_sample, np.dot(np.transpose(x_mat),t_mat)).reshape(1,2)
print(cov_sample)
print(mean_sample)
plot_2d_gaussian(mean_sample, cov_sample)

mean_sample = mean_sample.reshape(2)
weights = np.array([np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample), np.random.multivariate_normal(mean_sample, cov_sample)])

# print(weights)
for i in range(8):
	t_approx = weights[i][0]*x + weights[i][1]*np.ones(200)
	plt.plot(x, t_approx)
for i in range(10):
	plt.plot(x_mat[i][0], t_mat[i][0], 'ro')
	print()

plt.xlabel('input x')
plt.ylabel('output t')
plt.title('Functions drawn from the posterior, given ten data points')
plt.show()
'''

