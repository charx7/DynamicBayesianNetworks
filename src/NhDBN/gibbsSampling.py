import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma
from tqdm import tqdm
from utils import generateData, constructDesignMatrix, constructMuMatrix
from plotData import plotTrace, plotHistogram, plotScatter

def VanillaGibbsSampling(data, numSamples, numIter = 9000):
  # Design Matrix
  X = constructDesignMatrix(data, numSamples)
  # Retrieve the response vector
  y = data['response']['y']

  # Num Features
  numFeatures = len(data['features'])
  X_cols = X.shape[1] 

  # Initialization of the Gibbs Sampling
  pi = np.ones(numFeatures) # In this case pi is all ones (We are using all the features)
  beta = []
  sigma_sqr = [] # noise variance parameter
  lambda_sqr = []
  T = numSamples # T is the number of data points
  mu = constructMuMatrix(pi) # Prior expectation of betas is the 0 vector
  # Append the initial values of the vectors
  beta.append(np.zeros(X_cols))
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
  for it in tqdm(range(numIter)):
    ################# 1(a) Get a sample from sigma square
    el1 = (y.reshape(numSamples, 1) -  np.dot(X, mu)).T
    el2 = np.linalg.inv(np.identity(numSamples) + lambda_sqr[it] * np.dot(X, X.T))
    el3 = (y.reshape(numSamples, 1) -  np.dot(X, mu))

    # Gamma function parameters
    a_gamma = alpha_gamma_sigma_sqr + (T/2)
    b_gamma = np.asscalar(beta_gamma_sigma_sqr + 0.5 * (np.dot(np.dot(el1 ,el2),el3)))

    # Sample from the inverse gamma using the parameters and append to the vector of results
    #curr_sigma_sqr = 1 / (np.random.gamma(a_gamma, b_gamma)) #Not the correct Dist to sample
    curr_sigma_sqr = 1 / (invgamma.rvs(a_gamma, scale = b_gamma, size = 1))
    sigma_sqr.append(np.asscalar(curr_sigma_sqr))

    ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
    # Mean Vector Calculation
    el1 = np.linalg.inv(((1/(lambda_sqr[it])) * np.identity(X_cols)) + np.dot(X.T, X))
    el2 = ((1/(lambda_sqr[it])) * mu) + np.dot(X.T, y.reshape(100,1))
    curr_mean_vector = np.dot(el1, el2)
    # Sigma vector Calculation
    curr_cov_matrix = sigma_sqr[it + 1] * np.linalg.inv(((1/lambda_sqr[it]) * np.identity(X_cols) + np.dot(X.T, X)))
    sample = np.random.multivariate_normal(curr_mean_vector.flatten(), curr_cov_matrix)
    # Append the sample
    beta.append(sample)

    ################ 3(a) Get a sample of lambda square from a Gamma distribution
    el1 = np.dot((beta[it + 1] - mu.flatten()).reshape(X_cols,1).T, (beta[it + 1] - mu.flatten()).reshape(X_cols,1))  
    el2 = ((1/2) * (1/sigma_sqr[1]))
    a_gamma = alpha_gamma_lambda_sqr + ((X.shape[1])/2)
    b_gamma = beta_gamma_lambda_sqr + el2 * el1
    sample = 1/(invgamma.rvs(a_gamma, scale= b_gamma))
    # Append the sampled value
    lambda_sqr.append(sample)

  return {
    'lambda_sqr_vector': lambda_sqr,
    'sigma_sqr_vector': sigma_sqr,
    'betas_vector': beta
  }

def testAlgorithm():
  # Set Seed
  np.random.seed(42)
  # Generate data to test our algo
  num_samples = 100
  data = generateData(num_samples = num_samples, dimensions = 3, dependent = 1)
  
  # Do the gibbs Sampling
  results = VanillaGibbsSampling(data, num_samples)
  print('I have finished running the gibbs sampler!')
  
  # Plot Results for a particular \beta
  # List comprehension to get for some beta
  beta_0 = [x[1] for x in results['betas_vector']]
  plotHistogram(beta_0, 'beta_0', 1000)
  plotTrace(beta_0, 'beta_0', 1000)

  # Plot results for lambda^2
  plotHistogram(results['lambda_sqr_vector'], 'lambda^2', 1000)
  plotTrace(results['lambda_sqr_vector'], 'lambda^2', 1000)

  # Plot results for sigma^2
  plotHistogram(results['sigma_sqr_vector'], 'sigma^2', 1000)
  plotTrace(results['sigma_sqr_vector'], 'sigma^2', 1000)


if __name__ == '__main__':
  testAlgorithm()  
