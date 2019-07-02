import numpy as np
from random import randint
from tqdm import tqdm
from utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, \
  deleteMove, addMove, exchangeMove, selectData, constructNdArray
from scores import calculateFeatureScores
from marginalLikelihood import calculateMarginalLikelihood
from priors import calculateFeatureSetPriorProb

def pwGibbsSamplingWithMoves(data, changePoints, numSamples, numIter = 5000):
  # Initialization of the Gibbs Sampling
  fanInRestriction = 3
  featureDimensionSpace = len(dict.keys(data['features']))
  pi = [1] # Start with an empty feature set TODO remove the [1] ...was for testing
  partialData = selectData(data, pi) # Select just the columns according to the feature-set 
  
  # Partition data into each cp
  X = constructNdArray(partialData, numSamples, changePoints)
  # Retrieve the response vector
  y = data['response']['y'] # TODO we have to partition y for each changepoint as well

  # Get the amount of columns on the current design matrix
  X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes

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
