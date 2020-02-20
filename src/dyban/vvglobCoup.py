import numpy as np
from tqdm import tqdm 
from .utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, \
  deleteMove, addMove, exchangeMove, selectData, constructNdArray, constructResponseNdArray
from .samplers import segmentSigmaSampler, vvBetaSamplerWithChangepoints, \
  lambdaSqrSamplerWithChangepoints, vvMuSampler
from .moves import vvGlobCoupPiMove, vvGlobCoupTauMove

from .bayesianPwLinearRegression import BayesianPieceWiseLinearRegression

class VVglobCoupled(BayesianPieceWiseLinearRegression):
  def __init__(self, data, _type, num_samples, num_iter, change_points):
    # Call the contructor of the class BpwLinReg 
    super().__init__(data, _type, num_samples, num_iter, change_points)  
  
  def fit(self):
    '''
      Method that will produce the output of the chain of mcmc samples from 
      the posterior distribution using an MCMC
    '''
    # Initialization of the Gibbs Sampling
    fanInRestriction = 3
    T = self.num_samples # T is the number of data points
    featureDimensionSpace = len(dict.keys(self.data['features']))
    pi = [] # Start with an empty feature set 
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
    padded_betas = []
    sigma_sqr = [] # noise variance parameter
    sigma_sqr_vector = [] # TODO:merge with sigma_sqr? segment specific noise variance parameters
    lambda_sqr = []
    changePoints = self.change_points
    muVector = [] # vector of the \mu(s)

    # Append the initial values of the vectors
    selectedFeatures.append(pi)
    beta.append([np.zeros(len(pi) + 1)]) 
    padded_betas.append(np.zeros(featureDimensionSpace + 1)) # for the padded betas vector
    sigma_sqr.append(1) # (OLD) -> TODO maybe remove this 
    sigma_sqr_vector.append([1]) # append the first element 
    lambda_sqr.append(1)
    muVector.append(mu)

    # Main for loop of the gibbs sampler
    for it in tqdm(range(self.num_iter)):
      ################# 1(b) Get a sample from sigma square
      curr_sigma_sqr = segmentSigmaSampler(y, X, muVector[it], lambda_sqr[it],
        alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, self.num_samples, changePoints)
      # Append to the sigma vector
      sigma_sqr_vector.append(curr_sigma_sqr)

      ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
      sample = vvBetaSamplerWithChangepoints(y, X, muVector[it], 
        lambda_sqr[it], sigma_sqr_vector[it + 1], X_cols, self.num_samples, T, it, changePoints)
      # Append the sample
      beta.append(sample)
      padded_sample = self.transform_beta_coef(sample, pi, featureDimensionSpace)
      padded_betas.append(padded_sample)

      ################ 3(a) Get a sample of lambda square from a Gamma distribution
      sample = lambdaSqrSamplerWithChangepoints(X, beta[it + 1], muVector[it], 
        sigma_sqr_vector[it + 1], X_cols, alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr,
        changePoints)
      # Append the sampled value
      lambda_sqr.append(np.asscalar(sample))
      
      ################ Propose a move from pi -> pi* and \mu -> \mu*
      pi, currMu, X = vvGlobCoupPiMove(self.data, X, y, muVector[it],
       alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
       lambda_sqr, sigma_sqr_vector[it + 1], pi, fanInRestriction, featureDimensionSpace,
       self.num_samples, it, changePoints)
      
      # Append to the vector of results
      selectedFeatures.append(pi)
      muVector.append(currMu)

      # ---> Reconstruct the design ndArray, mu vector and parameters for the next iteration
      # Select the data according to the set Pi or Pi*
      partialData = selectData(self.data, pi)
      # Design ndArray
      X = constructNdArray(partialData, self.num_samples, changePoints)
      respVector = self.data['response']['y'] # We have to partition y for each changepoint as well
      y = constructResponseNdArray(respVector, changePoints)
      # Mu matrix
      mu = constructMuMatrix(pi)
      
      ################ Propose a change in the changepoints from tau to tau*
      changePoints = vvGlobCoupTauMove(self.data, X, y, muVector[it + 1],
        alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], sigma_sqr_vector[it + 1],
        pi, self.num_samples, changePoints)

      selectedChangepoints.append(changePoints) # append the selected cps
      
      # segment X and Y according the new change points
      partialData = selectData(self.data, pi)
      # Design ndArray
      X = constructNdArray(partialData, self.num_samples, changePoints)
      respVector = self.data['response']['y'] # We have to partition y for each changepoint as well
      y = constructResponseNdArray(respVector, changePoints)

      # Get the new column size of the design matrix
      X_cols = [cp.shape[1] for cp in X] # This is now a vector of shapes
      
    self.results = {
      'lambda_sqr_vector': lambda_sqr,
      'sigma_sqr_vector': sigma_sqr_vector,
      'pi_vector': selectedFeatures,
      'tau_vector': selectedChangepoints,
      'padded_betas': padded_betas
    }
