import numpy as np
from random import randint
from tqdm import tqdm
from utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, \
  deleteMove, addMove, exchangeMove, selectData, constructNdArray, constructResponseNdArray
from scores import calculateFeatureScores
from marginalLikelihood import calculateMarginalLikelihood
from priors import calculateFeatureSetPriorProb
from samplers import sigmaSqrSamplerWithChangePoints, betaSamplerWithChangepoints, \
  lambdaSqrSamplerWithChangepoints
from moves import featureSetMoveWithChangePoints, changepointsSetMove

def pwGibbsSamplingWithCpsParentsMoves(data, changePoints, numSamples, numIter = 5000):
  # Initialization of the Gibbs Sampling
  fanInRestriction = 3
  T = numSamples # T is the number of data points
  featureDimensionSpace = len(dict.keys(data['features']))
  pi = [] # Start with an empty feature set TODO remove the [1] ...was for testing
  partialData = selectData(data, pi) # Select just the columns according to the feature-set 
  
  # Partition data into each cp
  X = constructNdArray(partialData, numSamples, changePoints)
  # Retrieve the response vector
  respVector = data['response']['y'] # We have to partition y for each changepoint as well
  y = constructResponseNdArray(respVector, changePoints)
  
  # Get the amount of columns on the current design matrix
  X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes

  mu = constructMuMatrix(pi) # TODO this also has to be a vector, prior expectation is the zero vector

  # Use a collapsed sampler gibbs sampler \beta is integrated out with GAM ~ (a,b)
  # Standard choice of hyperparameters for lambda^2
  alpha_gamma_lambda_sqr = 2
  beta_gamma_lambda_sqr = 0.2
  # Standard choice of hyperparameters for sigma^2
  alpha_gamma_sigma_sqr = 0.01
  beta_gamma_sigma_sqr = 0.01
  
  selectedFeatures = [] # Empty initial parent set
  selectedChangepoints = [] # Empty initial changepoints set
  beta = []
  sigma_sqr = [] # noise variance parameter
  lambda_sqr = []
  
  # Append the initial values of the vectors
  selectedFeatures.append(pi)
  beta.append(np.zeros(len(pi) + 1)) # TODO this beta should be a dict
  sigma_sqr.append(1)
  lambda_sqr.append(1)

  # Main for loop of the gibbs sampler
  for it in tqdm(range(numIter)):
    ################# 1(b) Get a sample from sigma square
    curr_sigma_sqr = sigmaSqrSamplerWithChangePoints(y, X, mu, lambda_sqr,
     alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, numSamples, T, it, changePoints)
    # Append to the sigma vector
    sigma_sqr.append(np.asscalar(curr_sigma_sqr))

    ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
    sample = betaSamplerWithChangepoints(y, X, mu, 
      lambda_sqr, sigma_sqr, X_cols, numSamples, T, it, changePoints)
    # Append the sample
    beta.append(sample)

    ################ 3(a) Get a sample of lambda square from a Gamma distribution
    sample = lambdaSqrSamplerWithChangepoints(X, beta, mu, sigma_sqr, X_cols,
      alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr, it, changePoints)
    # Append the sampled value
    lambda_sqr.append(np.asscalar(sample))

    ################ 4(b) This step proposes a change on the feature set Pi to Pi*
    pi = featureSetMoveWithChangePoints(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
      lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples, it, changePoints)
    # Append to the vector of results
    selectedFeatures.append(pi)

    ################ 5(c) This step will propose a change in the changepoints from tau to tau*
    changePoints = changepointsSetMove(data, X, y, mu, alpha_gamma_lambda_sqr, beta_gamma_sigma_sqr,
      lambda_sqr, pi, numSamples, it, changePoints)

    # ---> Reconstruct the design ndArray, mu vector and parameters for the next iteration
    # Select the data according to the set Pi or Pi*
    partialData = selectData(data, pi)
    # Design ndArray
    X = constructNdArray(partialData, numSamples, changePoints)
    respVector = data['response']['y'] # We have to partition y for each changepoint as well
    y = constructResponseNdArray(respVector, changePoints)
    # Mu matrix
    mu = constructMuMatrix(pi)
    # Get the new column size of the design matrix
    X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes
  
  return {
    'lambda_sqr_vector': lambda_sqr,
    'sigma_sqr_vector': sigma_sqr,
    'pi_vector': selectedFeatures,
    'tau_vector': selectedChangepoints
  }
      
def pwGibbsSamplingWithMoves(data, changePoints, numSamples, numIter = 5000):
  # Initialization of the Gibbs Sampling
  fanInRestriction = 3
  T = numSamples # T is the number of data points
  featureDimensionSpace = len(dict.keys(data['features']))
  pi = [] # Start with an empty feature set TODO remove the [1] ...was for testing
  partialData = selectData(data, pi) # Select just the columns according to the feature-set 
  
  # Partition data into each cp
  X = constructNdArray(partialData, numSamples, changePoints)
  # Retrieve the response vector
  y = data['response']['y'] # We have to partition y for each changepoint as well
  y = constructResponseNdArray(y, changePoints)
  
  # Get the amount of columns on the current design matrix
  X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes

  mu = constructMuMatrix(pi) # TODO this also has to be a vector, prior expectation is the zero vector

  # Use a collapsed sampler gibbs sampler \beta is integrated out with GAM ~ (a,b)
  # Standard choice of hyperparameters for lambda^2
  alpha_gamma_lambda_sqr = 2
  beta_gamma_lambda_sqr = 0.2
  # Standard choice of hyperparameters for sigma^2
  alpha_gamma_sigma_sqr = 0.01
  beta_gamma_sigma_sqr = 0.01
  
  selectedFeatures = []
  beta = []
  sigma_sqr = [] # noise variance parameter
  lambda_sqr = []

  # Append the initial values of the vectors
  selectedFeatures.append(pi)
  beta.append(np.zeros(len(pi) + 1)) # TODO this beta should be a dict
  sigma_sqr.append(1)
  lambda_sqr.append(1)

  # Main for loop of the gibbs sampler
  for it in tqdm(range(numIter)):
    ################# 1(b) Get a sample from sigma square
    curr_sigma_sqr = sigmaSqrSamplerWithChangePoints(y, X, mu, lambda_sqr,
     alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, numSamples, T, it, changePoints)
    # Append to the sigma vector
    sigma_sqr.append(np.asscalar(curr_sigma_sqr))

    ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
    sample = betaSamplerWithChangepoints(y, X, mu, 
      lambda_sqr, sigma_sqr, X_cols, numSamples, T, it, changePoints)
    # Append the sample
    beta.append(sample)

    ################ 3(a) Get a sample of lambda square from a Gamma distribution
    sample = lambdaSqrSamplerWithChangepoints(X, beta, mu, sigma_sqr, X_cols,
      alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr, it, changePoints)
    # Append the sampled value
    lambda_sqr.append(np.asscalar(sample))

    ################ 4(b) This step proposes a change on the feature set Pi to Pi*
    pi = featureSetMoveWithChangePoints(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
      lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples, it, changePoints)
    # Append to the vector of results
    selectedFeatures.append(pi)

    ################ Reconstruct the design ndArray, mu vector and parameters for the next iteration
    # Select the data according to the set Pi or Pi*
    partialData = selectData(data, pi)
    # Design ndArray
    X = constructNdArray(partialData, numSamples, changePoints)
    # Mu matrix
    mu = constructMuMatrix(pi)
    # Get the new column size of the design matrix
    X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes
  
  return {
    'lambda_sqr_vector': lambda_sqr,
    'sigma_sqr_vector': sigma_sqr,
    'pi_vector': selectedFeatures
  }
  