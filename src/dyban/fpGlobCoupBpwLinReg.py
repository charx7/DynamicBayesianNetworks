import numpy as np
from tqdm import tqdm 
from .utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, \
  deleteMove, addMove, exchangeMove, selectData, constructNdArray, constructResponseNdArray
from .samplers import sigmaSqrSamplerWithChangePoints, betaSamplerWithChangepoints, \
  lambdaSqrSamplerWithChangepoints
from .moves import globCoupFeatureSetMoveWithChangePoints, globCoupChangepointsSetMove

from .bayesianPwLinearRegression import BayesianPieceWiseLinearRegression

class FpGlobCoupledBayesianPieceWiseLinearRegression(BayesianPieceWiseLinearRegression):
  def __init__(self, data, _type, num_samples, num_iter, change_points):
    # Call the contructor of the class BpwLinReg 
    super().__init__(data, _type, num_samples, num_iter, change_points)  
  
  def fit(self):
    '''
      Method that will produce the output of the chain of mcmc samples from 
      the posterior distribution using an MCMC
    '''
    # Initialization of the Gibbs Sampling
    fanInRestriction = 3 # in the fp model we should not have a fan in restriction
    T = self.num_samples # T is the number of data points
    featureDimensionSpace = len(dict.keys(self.data['features']))

    # set up full parents for the current network configuration
    parents = list(dict.keys(self.data['features']))
    parents = [int(string.lstrip('X')) for string in parents]
    parents = sorted(parents)

    pi = parents # Start with the full-parent set 
    partialData = selectData(self.data, pi) # Select just the columns according to the feature-set 
    
    # Partition data into each cp
    X = constructNdArray(partialData, self.num_samples, self.change_points)
    # Retrieve the response vector
    respVector = self.data['response']['y'] # We have to partition y for each changepoint as well
    y = constructResponseNdArray(respVector, self.change_points)
    
    # Get the amount of columns on the current design matrix
    X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes

    mu = constructMuMatrix(pi) # prior expectation is the zero vector

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
    changePoints = self.change_points
    muVector = [] # vector of the \mu(s)

    # Append the initial values of the vectors
    selectedFeatures.append(pi)
    beta.append([np.zeros(len(pi) + 1)]) # TODO this beta should be a dict
    sigma_sqr.append(1)
    lambda_sqr.append(1)
    muVector.append(mu)

    # Main for loop of the gibbs sampler
    for it in tqdm(range(self.num_iter)):
      ################# 1(b) Get a sample from sigma square
      curr_sigma_sqr = sigmaSqrSamplerWithChangePoints(y, X, muVector[it], lambda_sqr,
      alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, self.num_samples, T, it, changePoints)
      # Append to the sigma vector
      sigma_sqr.append(np.asscalar(curr_sigma_sqr))

      ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
      sample = betaSamplerWithChangepoints(y, X, muVector[it], 
        lambda_sqr, sigma_sqr, X_cols, self.num_samples, T, it, changePoints)
      # Append the sample
      beta.append(sample)

      ################ 3(a) Get a sample of lambda square from a Gamma distribution
      sample = lambdaSqrSamplerWithChangepoints(X, beta, muVector[it], sigma_sqr, X_cols,
        alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr, it, changePoints)
      # Append the sampled value
      lambda_sqr.append(np.asscalar(sample))

      # Check if the type is non-homgeneous to do inference over all possible cps
      if self._type == 'glob_coup_nh':  
        ################ 5(c) This step will propose a change in the changepoints from tau to tau*
        changePoints, currMu = globCoupChangepointsSetMove(self.data, X, y, muVector[it],
         alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr, sigma_sqr,
        pi, self.num_samples, it, changePoints)

        # Append to the vector of results
        muVector.append(currMu)

        # append to the tau vector of changepoints
        selectedChangepoints.append(changePoints)
        
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
      'tau_vector': selectedChangepoints,
      'mu_vector': muVector,
      'betas_vector': beta
    }
