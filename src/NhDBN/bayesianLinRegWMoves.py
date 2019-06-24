import numpy as np
from scipy.stats import invgamma
from random import randint
from tqdm import tqdm
from utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, deleteMove, addMove, exchangeMove, selectData
from scores import calculateFeatureScores, drawRoc
from samplers import sigmaSqrSampler, betaSampler, lambdaSqrSampler
from moves import featureSetMove
from marginalLikelihood import calculateMarginalLikelihood
from priors import calculateFeatureSetPriorProb
from plotData import plotTrace, plotHistogram, plotScatter
from generateTestData import generateTestDataSecond

def gibbsSamplingWithMoves(data, numSamples, numIter = 5000):
  # Initialization of the Gibbs Sampling
  # Uncomment if you want random initialization
  #pi = generateInitialFeatureSet(len(data['features']) + 1, 3)
  fanInRestriction = 3
  featureDimensionSpace = len(dict.keys(data['features']))
  pi = []
  partialData = selectData(data, pi) 
  
  # Design Matrix
  X = constructDesignMatrix(partialData, numSamples)
  # Retrieve the response vector
  y = data['response']['y']

  # Get the amount of columns on the current design matrix
  X_cols = X.shape[1] 

  selectedFeatures = []
  beta = []
  sigma_sqr = [] # noise variance parameter
  lambda_sqr = []
  T = numSamples # T is the number of data points
  mu = constructMuMatrix(pi) # Prior expectation is the zero vector
  
  # Append the initial values of the vectors
  selectedFeatures.append(pi)
  beta.append(np.zeros(len(pi) + 1)) # TODO this beta should be a dict
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
    curr_sigma_sqr = sigmaSqrSampler(y, X, mu, lambda_sqr, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, numSamples, T, it)
    sigma_sqr.append(np.asscalar(curr_sigma_sqr))

    ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
    sample = betaSampler(y, X, mu, lambda_sqr, sigma_sqr, X_cols, numSamples, T, it)
    # Append the sample
    beta.append(sample)

    ################ 3(a) Get a sample of lambda square from a Gamma distribution
    sample = lambdaSqrSampler(X, beta, mu, sigma_sqr, X_cols, alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr, it)
    # Append the sampled value
    lambda_sqr.append(np.asscalar(sample))

    ################ 4(a) This step proposes a change on the feature set Pi to Pi*
    pi = featureSetMove(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
      lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples, it)
    # Append to the vector of results
    selectedFeatures.append(pi)

    ################ Reconstruct the design matrix, mu vector and parameters for the next iteration
    # Select the data according to the set Pi or Pi*
    partialData = selectData(data, pi)
    # Design Matrix
    X = constructDesignMatrix(partialData, numSamples)
    # Mu matrix
    mu = constructMuMatrix(pi)
    # Get the new column size of the design matrix
    X_cols = X.shape[1] 
    
  return {
    'lambda_sqr_vector': lambda_sqr,
    'sigma_sqr_vector': sigma_sqr,
    'pi_vector': selectedFeatures
  }

def testAlgorithm():
  # Set Seed
  np.random.seed(42)
  # Generate data to test our algo
  num_samples = 100
  dims = 6
  data = generateTestDataSecond(num_samples = num_samples, dimensions = dims)

  # Do the gibbs Sampling
  results = gibbsSamplingWithMoves(data, num_samples)
  print('I have finished running the gibbs sampler!')
  res = calculateFeatureScores(results['pi_vector'][:3000], dims) 
  # Draw the RoC curve
  realEdges = {
    'X1': 0,
    'X2': 1,
    'X3': 0,
    'X4': 0,
    'X5': 1,
    'X6': 0
  }

  # Get the edge scores and compare
  y_score = list(res.values())
  y_real = list(realEdges.values())
  # Draw the RoC curve
  drawRoc(y_score, y_real)

if __name__ == '__main__':
  testAlgorithm()
