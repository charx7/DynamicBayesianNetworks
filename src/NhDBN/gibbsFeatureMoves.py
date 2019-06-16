import numpy as np
from scipy.stats import invgamma
from random import randint
from tqdm import tqdm
from utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, deleteMove, addMove, exchangeMove, selectData
from scores import calculateFeatureScores
from marginalLikelihood import calculateMarginalLikelihood
from priors import calculateFeatureSetPriorProb
from plotData import plotTrace, plotHistogram, plotScatter
from generateTestData import generateTestDataSecond

def gibbsSamplingWithMoves(data, numSamples, numIter = 9000):
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
    el1 = (y.reshape(numSamples, 1) -  np.dot(X, mu)).T
    el2 = np.linalg.inv(np.identity(numSamples) + lambda_sqr[it] * np.dot(X, X.T))
    el3 = (y.reshape(numSamples, 1) -  np.dot(X, mu))

    # Gamma function parameters
    a_gamma = alpha_gamma_sigma_sqr + (T/2)
    b_gamma = np.asscalar(beta_gamma_sigma_sqr + 0.5 * (np.dot(np.dot(el1 ,el2),el3)))

    # Sample from the inverse gamma using the parameters and append to the vector of results
    #curr_sigma_sqr = 1 / (np.random.gamma(a_gamma, b_gamma)) #Not the correct Dist to sample
    curr_sigma_sqr = 1 / (np.random.gamma(a_gamma, scale = (1 / b_gamma), size = 1))
    sigma_sqr.append(np.asscalar(curr_sigma_sqr))

    ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
    # Mean Vector Calculation
    el1 = np.linalg.inv(((1/(lambda_sqr[it])) * np.identity(X_cols)) + np.dot(X.T, X))
    el2 = ((1/(lambda_sqr[it])) * mu) + np.dot(X.T, y.reshape(numSamples, 1))
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
    sample = 1/(np.random.gamma(a_gamma, scale= (1/ b_gamma)))
    # Append the sampled value
    lambda_sqr.append(np.asscalar(sample))

    ################ 4(a) This step proposes a change on the feature set Pi to Pi*
    # Calculate the probability of response given the feature set Pi (marginal likelihood)
    marginalPi = calculateMarginalLikelihood(X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples)    
    # Select a random add, delete or exchange move
    randomInteger = randint(0,2)
    
    func = selectMoveDict(randomInteger)
    # Try catch block for the random move
    try:
      piStar = func(pi, featureDimensionSpace, fanInRestriction)
      # Construct the new X, mu
      partialData = {
        'features':{},
        'response':{}
      }
      for feature in piStar:
        currKey = 'X' + str(int(feature))
        partialData['features'][currKey] = data['features'][currKey]
  
      # Design Matrix
      XStar = constructDesignMatrix(partialData, numSamples)
      # Mu matrix
      muStar = constructMuMatrix(piStar)
      # Calculate marginal likelihook for PiStar
      marginalPiStar = calculateMarginalLikelihood(XStar, y, muStar, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples) 
      
    except ValueError:
      piStar = pi
      marginalPiStar = marginalPi
      # Calculate the probability of response the feature set Pi*
    
    # Calculate the prior probabilites of the move Pi -> Pi*
    piPrior = calculateFeatureSetPriorProb(pi, featureDimensionSpace, fanInRestriction) 
    piStarPrior = calculateFeatureSetPriorProb(piStar, featureDimensionSpace, fanInRestriction)

    # Calculate the acceptance/rejection probability of the move given Pi, Pi*
    # First we need to calculate HR given the move we selected
    if randomInteger == 0:
      # Add move
      hr = (featureDimensionSpace - len(pi)) / len(piStar)
    elif randomInteger == 1:
      # Delete Move
      hr = len(pi) / (featureDimensionSpace - len(piStar))
    elif randomInteger == 2:
      # Exchange move
      hr = 1
    # Get the threshhold of the probability of acceptance of the move
    acceptanceRatio = min(1, (marginalPiStar/marginalPi) * (piStarPrior/ piPrior) * hr)
    # Get a sample from the U(0,1) to compare the acceptance ratio
    u = np.random.uniform(0,1)
    if u < acceptanceRatio:
      # if the sample is less than the acceptance ratio we accept the move to Pi*
      pi = piStar

    # Select the data according to the set Pi or Pi*
    partialData = selectData(data, pi)
    
    # Design Matrix
    X = constructDesignMatrix(partialData, numSamples)
    # Mu matrix
    mu = constructMuMatrix(pi)
    # Get the new column size of the design matrix
    X_cols = X.shape[1] 
    # Append to the vector of results
    selectedFeatures.append(pi)

  return {
    'lambda_sqr_vector': lambda_sqr,
    'sigma_sqr_vector': sigma_sqr,
    'pi_vector': selectedFeatures
  }

# Switcher that defines what random move we are going to make
def selectMoveDict(selectedFunc):
  switcher = {
    0: addMove,
    1: deleteMove,
    2: exchangeMove
  }

  return switcher.get(selectedFunc)

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
  calculateFeatureScores(results['pi_vector'], dims) 

if __name__ == '__main__':
  testAlgorithm()
