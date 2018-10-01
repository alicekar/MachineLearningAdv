# Author: Alice Karnsund and Elin Samuelsson
import numpy as np
from math import pi, exp, sqrt
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel


# Generate som test data for the 2 pictures
def generateData(N):
    X = np.linspace(-10, 10, N)
    # without noise
    T1 = (1 / np.absolute(X)) * np.sin(np.absolute(X))
    # with noise
    T2 = np.copy(T1)
    for t in range(len(T2)):
        error = np.random.normal(0, 0.2)
        T2[t] = T2[t] + error
    return (X, T1, T2)


# Linear spline kernel
def splineMatrix(Y, X):
    N = len(X)
    matrix = np.ones((len(Y), N + 1))
    # K_mn = K(x_m,x_n-1)    obs note m=n and n=m in paper
    for m in range(len(Y)):
        x_m = Y[m]
        for n in range(N):
            x_n = X[n]
            elem = 1 + x_m * x_n + x_m * x_n * min(x_m, x_n) - 0.5 * (x_m + x_n) * min(x_m, x_n) ** 2
            elem = elem + (min(x_m, x_n) ** 3) / 3
            matrix[m][n + 1] = elem
    return (matrix)


# Gaussian kernel
def RBFMatrix(Y, X):
    K = np.ones((len(Y), len(X) + 1))
    X = np.asmatrix(X).T
    Y = np.asmatrix(Y).T
    k = rbf_kernel(Y, X)
    K[:, 1:] = k
    print(np.shape(k))
    print(np.shape(K))
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
    print('Längd gamma:', len(gamma_bool))
    print('K:', np.shape(K))
    return (gamma, alpha, K, Xrv, mu, Trv)


# Updating relavant parameters and prunes. Returns the optimal parameter values
def updateParams(iterations, K, alpha_start, N, sigma2, T, X):
    # For T1 the number of RVs converges to 9

    alpha_new = alpha_start
    sigma2_new = sigma2
    Trv = np.copy(T)
    Xrv = X

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
        #numer = np.dot((T - np.dot(K,mu)),(T - np.dot(K,mu)))
        #sigma2_new = numer/(N-sum(gamma))

        # Relevance vectors
        print('Number of RVs:', len(Xrv))
    return (alpha_new, sigma2_new, mu, Sigma, Xrv, Trv)


# Make predictions of unseen x's, x is a vector or matrix
def predictions(X_relevance, mu_s, Sigma_s, sigma2_s, alpha_s, x):
    K = splineMatrix(x, X_relevance)
    mu_s = np.asmatrix(mu_s)
    mean = np.dot(mu_s, K.T)
    Covar = sigma2_s + np.dot(np.dot(K,Sigma_s),K.T)
    Covar = np.diag(Covar)
    t = np.random.normal(mean,Covar)
    t = t.reshape(-1)
    mean = np.asarray(mean)
    mean = mean.reshape(-1)
    variance = np.sqrt(Covar)
    return(t, mean, variance)


# Calculate max error 
def error(T_true, mean):
    error = T_true-mean
    error = np.absolute(error)
    max_error = max(error)
    rms_error = sqrt(sum(error**2)/len(error))

    return(max_error, rms_error)



# ---------------------------------- RUN section ---------------------------------
# Number of data points
N = 100
iterations = 2000
sigma2 = 0.01 ** 2
alpha_start = np.ones(N + 1)  # note a0
X, T1, T2 = generateData(N)
K = splineMatrix(X, X)
#K = RBFMatrix(X,X)

# Change depending on data set
T = T1

# Get optimal parameters and relevance vectors
alpha_s, sigma2_s, mu_s, Sigma_s, X_relevance, T_relevance = updateParams(iterations, K, alpha_start, N, sigma2, T, X)


# Predict new data
x = np.linspace(-10,10,100)
t, mean, variance = predictions(X_relevance, mu_s, Sigma_s, sigma2_s, alpha_s, x)


# Calculate error
max_error, rms_error = error(T1,mean)
print('Max error:', max_error)
print('RMS error:', rms_error)
print('sigma^2:', sigma2_s)


#------------------------------------- P L O T ---------------------------------------
# Correct function
plt.plot(X, T1, 'pink', label = 'True function')

# Plot relevance vectors
plt.plot(X_relevance, T_relevance, 'ro', label = str(len(X_relevance))+' Relevance vectors')


# ------------------------------ Data without noise ----------------------------
# Plot approximate function and points when training data is without noise
plt.plot(X, T1, 'ro', markersize=2, label = 'Data points')
plt.plot(x, mean , 'g--', label = 'Predictive function')
plt.legend()
plt.xlabel('Feature inputs, x')
plt.ylabel('Target values, t')
plt.title('RVM, approximation of the Sinc function')
plt.show()
'''
# ------------------------------- Data with noise ------------------------------
# Plot points, mean and variance for new data, when training data is with noise
upper = mean + variance
lower = mean - variance

plt.plot(X, T2, 'ro', markersize = 2, label = 'Data points')
plt.plot(x, mean , 'g--', label = 'Predictive mean')
plt.plot(x, upper, 'g', linewidth = 0.5, label = 'Predictive variance')
plt.plot(x, lower, 'g', linewidth = 0.5)
plt.fill_between(x, upper, lower, color='g', alpha=.1)
plt.legend()
plt.xlabel('Feature inputs, x')
plt.ylabel('Target values, t')
plt.title('RVM, approximation of the Sinc function')

plt.show()



Max error: 0.170657690943
RMS error: 0.06327297039165751
sigma^2: 0.0296701959644

utan:
Max error: 0.00705348248013
RMS error: 0.002902341122605487
sigma^2: 0.0001
'''
