#from abc import ABC, abstractmethod # not sure if abc is worth the overhead cost
from tqdm import tqdm
import numpy as np

from .utils import constructDesignMatrix, constructMuMatrix, \
  deleteMove, addMove, exchangeMove, selectData, constructNdArray, constructResponseNdArray
from .samplers import sigmaSqrSampler, betaSampler, \
  lambdaSqrSampler
from .moves import featureSetMove

class BayesianLinearRegression:
  '''
    Base class for all the bayesian linear regression estimator algorithms
    
    Attributes:
      data : dict of the current configuration of the data
        {
          'features':{
            'X1': numpy.ndarray,
          }
          'response':{
            'y': numpy.ndarray
          }
        }
      num_samples : int
        number of samples on the data
      num_iter : int
        number of iterations
      results : dict of str: list<float>
        dictionary containing the results of the chain 
  '''
  # Base class for the bayesian linear regression class
  def __init__(self, data, num_samples, num_iter = 5000):
    self.data = data
    self.num_samples = num_samples
    self.num_iter = num_iter
    self.results = None

  # @abstractmethod
  def fit(self):
    '''
      Method that will Produce the output of the chain of mcmc samples from 
      the posterior distribution using an MCMC
    '''
    # Initialization of the Gibbs Sampling
    fanInRestriction = 3
    featureDimensionSpace = len(dict.keys(self.data['features']))
    pi = []
    partialData = selectData(self.data, pi) 
    
    # Design Matrix
    #change_points = [self.num_samples] # We have no changepoints since ints homogeneous data
    X = constructDesignMatrix(partialData, self.num_samples - 1)
    # Retrieve the response vector
    y = self.data['response']['y']

    # Get the amount of columns on the current design matrix
    X_cols = X.shape[1] 

    selectedFeatures = []
    beta = []
    sigma_sqr = [] # noise variance parameter
    lambda_sqr = []
    T = self.num_samples # T is the number of data points
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
    # Standard choice of hyperparameters for sigma^2
    alpha_gamma_sigma_sqr = 0.01
    beta_gamma_sigma_sqr = 0.01

    # Main for loop of the gibbs sampler
    for it in tqdm(range(self.num_iter)):
      ################# 1(a) Get a sample from sigma square
      curr_sigma_sqr = sigmaSqrSampler(y, X, mu, lambda_sqr, alpha_gamma_sigma_sqr, 
        beta_gamma_sigma_sqr, self.num_samples - 1, T, it)
      sigma_sqr.append(np.asscalar(curr_sigma_sqr))

      ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
      sample = betaSampler(y, X, mu, lambda_sqr, sigma_sqr, X_cols, self.num_samples - 1, T, it)
      # Append the sample
      beta.append(sample)

      ################ 3(a) Get a sample of lambda square from a Gamma distribution
      sample = lambdaSqrSampler(X, beta, mu, sigma_sqr, X_cols, alpha_gamma_lambda_sqr,
         beta_gamma_lambda_sqr, it)
      # Append the sampled value
      lambda_sqr.append(np.asscalar(sample))

      ################ 4(a) This step proposes a change on the feature set Pi to Pi*
      pi = featureSetMove(self.data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
        lambda_sqr, pi, fanInRestriction, featureDimensionSpace, self.num_samples - 1, it)
      # Append to the vector of results
      selectedFeatures.append(pi)

      ################ Reconstruct the design matrix, mu vector and parameters for the next iteration
      # Select the data according to the set Pi or Pi*
      partialData = selectData(self.data, pi)
      # Design Matrix
      X = constructDesignMatrix(partialData, self.num_samples - 1)
      # Mu matrix
      mu = constructMuMatrix(pi)
      # Get the new column size of the design matrix
      X_cols = X.shape[1] 
      
    self.results = {
      'lambda_sqr_vector': lambda_sqr,
      'sigma_sqr_vector': sigma_sqr,
      'pi_vector': selectedFeatures
    }
    