import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma

# Set Seed
np.random.seed(42)
# Generate data to test our algo
num_samples = 100
# Independent Variables
X1 = np.random.normal(0, 1, num_samples)
X2 = np.random.normal(0, 1, num_samples)
# Random noise to add to the dependent variable(s)
epsilon = np.random.normal(0, 0.1, num_samples)
# Dependent Variables
X3 = (X1 * -0.2 + X2 * 0.7) + epsilon
# Create the response vector y
y = np.random.normal(0, 1, num_samples)
# Cosntruct the design matrix X
ones_vector = np.ones(num_samples)
X = np.vstack((ones_vector, X1, X2, X3)).T

# Scatter Plot of the variables 
plt.scatter(X1, X3, alpha=0.5)
plt.title('Scatter plot of X1 and X3')
plt.xlabel('X1')
plt.ylabel('X3')
plt.show()

# Initialization of the Gibbs Sampling
pi = np.ones(3) # In this case pi is all ones and of size 3 because we have 3 features
beta = []
sigma_sqr = [] # noise variance parameter
lambda_sqr = []
numIter = 9000
T = num_samples # T is the number of data points
mu = np.zeros(4).reshape(4,1) # Not sure if this is correct prior exp of betas is 0 vector?
# Append the initial values of the vectors
beta.append(np.zeros(4))
sigma_sqr.append(1)
lambda_sqr.append(1)

# Use a collapsed sampler gibbs sampler \beta is integrated out with GAM ~ (a,b)
# Standard choice of hyperparameters for lambda^2
alpha_gamma_lambda_sqr = 2
beta_gamma_lambda_sqr = 0.2
# Stndard choice of hyperparameters for sigma^2
alpha_gamma_sigma_sqr = 0.01
beta_gamma_sigma_sqr = 0.01

# Main for loop of the gibbs sampler
for it in range(numIter):
    ################# 1(a) Get a sample from sigma square
    el1 = (y.reshape(num_samples, 1) -  np.dot(X, mu)).T
    el2 = np.linalg.inv(np.identity(num_samples) + lambda_sqr[it] * np.dot(X, X.T))
    el3 = (y.reshape(num_samples, 1) -  np.dot(X, mu))

    # Gamma function parameters
    a_gamma = alpha_gamma_sigma_sqr + (T/2)
    b_gamma = np.asscalar(beta_gamma_sigma_sqr + 0.5 * (np.dot(np.dot(el1 ,el2),el3)))

    # Sample from the inverse gamma using the parameters and append to the vector of results
    #curr_sigma_sqr = 1 / (np.random.gamma(a_gamma, b_gamma)) #Not the correct Dist to sample
    curr_sigma_sqr = 1 / (invgamma.rvs(a_gamma, scale = b_gamma, size = 1))
    sigma_sqr.append(np.asscalar(curr_sigma_sqr))

    ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
    # Mean Vector Calculation
    el1 = np.linalg.inv(((1/(lambda_sqr[it])) * np.identity(4)) + np.dot(X.T, X))
    el2 = ((1/(lambda_sqr[it])) * mu) + np.dot(X.T, y.reshape(100,1))
    curr_mean_vector = np.dot(el1, el2)
    # Sigma vector Calculation
    curr_cov_matrix = sigma_sqr[it + 1] * np.linalg.inv(((1/lambda_sqr[it]) * np.identity(4) + np.dot(X.T, X)))
    sample = np.random.multivariate_normal(curr_mean_vector.flatten(), curr_cov_matrix)
    # Append the sample
    beta.append(sample)

    ################ 3(a) Get a sample of lambda square from a Gamma distribution
    el1 = np.dot((beta[it + 1] - mu.flatten()).reshape(4,1).T, (beta[it + 1] - mu.flatten()).reshape(4,1))  
    el2 = ((1/2) * (1/sigma_sqr[1]))
    a_gamma = alpha_gamma_lambda_sqr + ((X.shape[1])/2)
    b_gamma = beta_gamma_lambda_sqr + el2 * el1
    sample = 1/(invgamma.rvs(a_gamma, scale= b_gamma))
    # Append the sampled value
    lambda_sqr.append(sample)

print('I have finished running the gibbs sampler!')
# Test plot for betas
# List comprehension to get for some beta
beta_0 = [x[1] for x in beta]
plt.hist(beta_0[1000:], bins='auto')
plt.show()

plt.plot(beta_0[1000:])
plt.show()

# Quick histogram of the betas vector to check the algo
plt.hist(lambda_sqr[1000:], bins='auto')
plt.title('Histogram of the Variable')
plt.show()

plt.plot(lambda_sqr[1000:])
plt.title('Gibbs sampler trace plot')
plt.show()
