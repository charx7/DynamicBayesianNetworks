import numpy as np
from tqdm import tqdm

from .bayesianLinearRegression import BayesianLinearRegression
from .utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, \
  deleteMove, addMove, exchangeMove, selectData, constructNdArray, constructResponseNdArray
from .samplers import sigmaSqrSamplerWithChangePoints, betaSamplerWithChangepoints, \
  lambdaSqrSamplerWithChangepoints
from .moves import featureSetMoveWithChangePoints, changepointsSetMove

class BayesianPieceWiseLinearRegression(BayesianLinearRegression):
  '''
    Bayesian Piece-wise Linear Regression algorithm with 
    varying changepoints and parent sets

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
        number of data points
      num_iter : int
        number of iterations of the MCMC algorithm
      change_points : list of int
        list containing the marked initial change points
      results : dict of string: list<float>
        dictionary containing the results of the sampled chain
  '''
  def __init__(self, data, _type, num_samples, num_iter, change_points):
    # Call the contructor of the base class BayesianLinearRegression
    super().__init__(data, num_samples, num_iter)
    self._type = _type
    self.change_points = change_points
  
  def fit(self):
    '''
      Method that will Produce the output of the chain of mcmc samples from 
      the posterior distribution using an MCMC
    '''
    # Initialization of the Gibbs Sampling
    fanInRestriction = 3
    T = self.num_samples # T is the number of data points
    featureDimensionSpace = len(dict.keys(self.data['features']))
    pi = [] # Start with an empty feature set TODO remove the [1] ...was for testing
    partialData = selectData(self.data, pi) # Select just the columns according to the feature-set 
    
    # Partition data into each cp
    X = constructNdArray(partialData, self.num_samples, self.change_points)
    # Retrieve the response vector
    respVector = self.data['response']['y'] # We have to partition y for each changepoint as well
    y = constructResponseNdArray(respVector, self.change_points)
    
    # Get the amount of columns on the current design matrix
    X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes

    mu = constructMuMatrix(pi) # TODO this also has to be a vector, prior expectation is the zero vector

    # Use a collapsed sampler gibbs sampler \beta is integrated out with GAM ~ (a,b)
    # Standard choice of hyperparameters for lambda^2
    alpha_gamma_lambda_sqr = 2
    beta_gamma_lambda_sqr = 0.2
    # Standard choice of hyperparameters for sigma^2
    alpha_gamma_sigma_sqr = 0.005
    beta_gamma_sigma_sqr = 0.005
    
    selectedFeatures = [] # Empty initial parent set
    selectedChangepoints = [] # Empty initial changepoints set
    beta = []
    sigma_sqr = [] # noise variance parameter
    lambda_sqr = []
    changePointsVector = []
    changePoints = self.change_points
    
    # Append the initial values of the vectors
    selectedFeatures.append(pi)
    beta.append([np.zeros(len(pi) + 1)]) # TODO this beta should be a dict
    sigma_sqr.append(1)
    lambda_sqr.append(1)

    # Main for loop of the gibbs sampler
    for it in tqdm(range(self.num_iter)):
      ################# 1(b) Get a sample from sigma square
      curr_sigma_sqr = sigmaSqrSamplerWithChangePoints(y, X, mu, lambda_sqr,
      alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, self.num_samples, T, it, changePoints)
      # Append to the sigma vector
      sigma_sqr.append((curr_sigma_sqr).item())

      ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
      sample = betaSamplerWithChangepoints(y, X, mu, 
        lambda_sqr, sigma_sqr, X_cols, self.num_samples, T, it, changePoints)
      # Append the sample
      beta.append(sample)

      ################ 3(a) Get a sample of lambda square from a Gamma distribution
      sample = lambdaSqrSamplerWithChangepoints(X, beta, mu, sigma_sqr, X_cols,
        alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr, it, changePoints)
      # Append the sampled value
      lambda_sqr.append((sample).item())

      ################ 4(b) This step proposes a change on the feature set Pi to Pi*
      pi, X, mu = featureSetMoveWithChangePoints(self.data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
        lambda_sqr, pi, fanInRestriction, featureDimensionSpace, self.num_samples, it, changePoints)
      # Append to the vector of results
      selectedFeatures.append(pi)

      # Check if the type is non-homgeneous to do inference over all possible cps
      if self._type == 'varying_nh':  
        ################ 5(c) This step will propose a change in the changepoints from tau to tau*
        changePoints = changepointsSetMove(self.data, X, y, mu, alpha_gamma_sigma_sqr,
          beta_gamma_sigma_sqr, lambda_sqr, pi, self.num_samples, it, changePoints)
        changePointsVector.append(changePoints) # for monitoring
      # ---> Reconstruct the design ndArray, mu vector and parameters for the next iteration
      # Select the data according to the set Pi or Pi*
      partialData = selectData(self.data, pi)
      # Design ndArray
      X = constructNdArray(partialData, self.num_samples, changePoints)
      respVector = self.data['response']['y'] # We have to partition y for each changepoint as well
      y = constructResponseNdArray(respVector, changePoints)
      # Mu matrix
      mu = constructMuMatrix(pi)
      # Get the new column size of the design matrix
      X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes
    
    self.results = {
      'lambda_sqr_vector': lambda_sqr,
      'sigma_sqr_vector': sigma_sqr,
      'pi_vector': selectedFeatures,
      'tau_vector': changePointsVector
    }
