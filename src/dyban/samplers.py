import numpy as np
from scipy.stats import multivariate_normal

def betaTildeSampler(y, X, mu, change_points, lambda_sqr, delta_sqr):
  '''
    Returns a vector of Posterior Beta expectations according to the segmentation
    of the data

    Args:
      y : list(numpy.ndarray(float))
       response list by changepoint
      X : list(numpy.ndarray(float))
        The whole data in a list by changepoint
      change_points : list(int)
        list of changepoints
      lambda_sqr : float
        signal to noise ratio hyper-parameter
      delta_sqr : float
        coupling strength hyper-parameter
    
    Returns:
      betaTilde : list(numpy.ndarray(float))
        vector of the sampled beta Tildes
  '''

  betaTilde = []
  X_h_before = None
  y_h_before = None
  cpLenBefore = None
  # Loop through all cps 
  for idx, _ in enumerate(change_points):
    currCplen = y[idx].shape[0] # Get the length of the cp
    y_h = y[idx] # Get the current sub y vector
    X_h = X[idx] # Get the current design matrix

    if idx == 1: # we are on the second cp
      el1 = np.linalg.inv(((1 / lambda_sqr) * np.identity(X_h.shape[1])) + np.dot(X_h.T, X_h))   
      el2 = np.dot(X_h.T, y_h.reshape(currCplen, 1))
      betaTilde.append(np.dot(el1, el2))
    elif idx > 1: # we are beyond the second cp
      el1 = np.linalg.inv(((1 / delta_sqr) * np.identity(X_h_before.shape[1])) \
        + np.dot(X_h_before.T, X_h_before))
      # TODO check the idx - 2 or 1 with marco   
      el2 = (1 / delta_sqr) * betaTilde[idx - 2] + np.dot(
        X_h_before.T, y_h_before.reshape(cpLenBefore, 1))
      betaTilde.append(np.dot(el1, el2))
    else: # we are on the first cp
      betaTilde.append(mu)
    # Save the prev segments for later use
    X_h_before = X_h
    y_h_before = y_h
    cpLenBefore = currCplen

  return betaTilde 

def sigmaSqrSamplerWithChangePointsSeqCop(y, X, mu, lambda_sqr, alpha_gamma_sigma_sqr, 
   beta_gamma_sigma_sqr, numSamples, T, it, change_points, 
   delta_sqr):
  # construct the betas obj posterior expectation for each cp
  betas = betaTildeSampler(y, X, mu, change_points, lambda_sqr[it], delta_sqr[it])

  ################# 1(b) Get a sample from sigma square
  h_prod_sum = 0 # The sum that will accumulate between each changepoint
  for idx, cp in enumerate(change_points):
    currCplen = y[idx].shape[0]
    y_h = y[idx] # Get the current sub y vector
    X_h = X[idx] # Get the current design matrix
    mu = betas[idx]

    # seq-coupled schema for the calculation of the C matrix
    if idx == 0: # We are on the first cp so the prior exp is 0
      el2 = np.linalg.inv(np.identity(currCplen) + lambda_sqr[it] * np.dot(X_h, X_h.T))
    else:
      el2 = np.linalg.inv(np.identity(currCplen) + delta_sqr[it] * np.dot(X_h, X_h.T))
    
    el1 = (y_h.reshape(currCplen, 1) - np.dot(X_h, mu)).T
    el3 = (y_h.reshape(currCplen, 1) -  np.dot(X_h, mu))

    h_prod_sum += np.dot(np.dot(el1, el2), el3) # accumulate the sum 

  # Gamma function parameters
  a_gamma = alpha_gamma_sigma_sqr + (T/2)
  b_gamma = np.asscalar(beta_gamma_sigma_sqr + 0.5 * (h_prod_sum))

  # Sample from the inverse gamma using the parameters and append to the vector of results
  curr_sigma_sqr = 1 / (np.random.gamma(a_gamma, scale = (1 / b_gamma), size = 1))

  return curr_sigma_sqr

def sigmaSqrSamplerWithChangePoints(y, X, mu, lambda_sqr, alpha_gamma_sigma_sqr, \
   beta_gamma_sigma_sqr, numSamples, T, it, change_points):
  ################# 1(b) Get a sample from sigma square
  h_prod_sum = 0 # The sum that will accumulate between each changepoint
  for idx, cp in enumerate(change_points):
    currCplen = y[idx].shape[0]
    y_h = y[idx] # Get the current sub y vector
    X_h = X[idx] # Get the current design matrix
    # TODO do the same logic for the \mu vector
    el1 = (y_h.reshape(currCplen, 1) - np.dot(X_h, mu)).T
    el2 = np.linalg.inv(np.identity(currCplen) + lambda_sqr[it] * np.dot(X_h, X_h.T))
    el3 = (y_h.reshape(currCplen, 1) -  np.dot(X_h, mu))
    
    h_prod_sum += np.dot(np.dot(el1, el2), el3) # accumulate the sum 

  # Gamma function parameters
  a_gamma = alpha_gamma_sigma_sqr + (T/2)
  #b_gamma = np.asscalar(beta_gamma_sigma_sqr + 0.5 * (h_prod_sum))
  b_gamma = (beta_gamma_sigma_sqr + 0.5 * (h_prod_sum)).item()

  # Sample from the inverse gamma using the parameters and append to the vector of results
  invSample = np.random.gamma(a_gamma, scale = ( 1 / b_gamma), size = 1)
  curr_sigma_sqr =  1 / invSample

  return curr_sigma_sqr

def sigmaSqrSampler(y, X, mu, lambda_sqr, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, numSamples, T, it):
  ################# 1(a) Get a sample from sigma square
  el1 = (y.reshape(numSamples, 1) -  np.dot(X, mu)).T
  el2 = np.linalg.inv(np.identity(numSamples) + lambda_sqr[it] * np.dot(X, X.T))
  el3 = (y.reshape(numSamples, 1) -  np.dot(X, mu))

  # Gamma function parameters
  a_gamma = alpha_gamma_sigma_sqr + (T/2)
  b_gamma = np.asscalar(beta_gamma_sigma_sqr + 0.5 * (np.dot(np.dot(el1, el2), el3)))

  # Sample from the inverse gamma using the parameters and append to the vector of results
  #curr_sigma_sqr = 1 / (np.random.gamma(a_gamma, b_gamma)) #Not the correct Dist to sample
  curr_sigma_sqr = 1/(np.random.gamma(a_gamma, scale = (1 / b_gamma), size = 1))
  
  return curr_sigma_sqr

def betaSamplerWithChangepointsSeqCoup(
  y, X, mu, lambda_sqr, sigma_sqr, delta_sqr, X_cols, numSamples, T, it, change_points):
  betasVector = [] # start with an empty betas vector
  # get the betas posterior expectations
  betasTilde = betaTildeSampler(y, X, mu, change_points, lambda_sqr[it], delta_sqr[it])

  for idx, _ in enumerate(change_points):
    currCplen = y[idx].shape[0] # length of the segment
    y_h = y[idx] # Get the current sub y vector
    X_h = X[idx] # Get the current design matrix
    X_cols_h = X_cols[idx] # Get the current cols of the current cp
    mu = betasTilde[idx] # Get the current betas posterior expectation
    
    if idx == 0: # we are on the first cp so we use the lambda hyperparam
      currSample = betaSampler(y_h, X_h, mu, lambda_sqr, 
        sigma_sqr, X_cols_h, currCplen, T, it)
    else: # on the next cps we use delta_sqr hyper param (coupling param)
      currSample = betaSampler(y_h, X_h, mu, delta_sqr,
        sigma_sqr, X_cols_h, currCplen, T, it)
    
    betasVector.append(currSample) # append to the betas vector

  return betasVector # return the constructed betas vector

def betaSamplerWithChangepoints(y, X, mu, lambda_sqr, sigma_sqr, X_cols, numSamples, T, it, change_points):
  betasVector = []
  ################ 2(b) Get a sample form the beta multivaratiate Normal for each cp
  for idx, cp in enumerate(change_points):
    currCplen = y[idx].shape[0]
    y_h = y[idx] # Get the current sub y vector
    X_h = X[idx] # Get the current design matrix
    # TODO we have to get \mu using the same logic (per cp)
    X_cols_h = X_cols[idx] # Get the current cols of the current cp

    # Get the sample from beta using teh betaSampler func
    currCpBeta = betaSampler(y_h, X_h, mu, 
      lambda_sqr, sigma_sqr, X_cols_h, currCplen, T, it)
    # Append to the betas vector list
    betasVector.append(currCpBeta)

  return betasVector

def betaSampler(y, X, mu, lambda_sqr, sigma_sqr, X_cols, numSamples, T, it):
  ################ 2(a) Get a sample of Beta form the multivariate Normal distribution
  # Mean Vector Calculation
  el1 = np.linalg.inv(((1/(lambda_sqr[it])) * np.identity(X_cols)) + np.dot(X.T, X))
  el2 = ((1/(lambda_sqr[it])) * mu) + np.dot(X.T, y.reshape(numSamples, 1))
  curr_mean_vector = np.dot(el1, el2)
  # Sigma vector Calculation
  curr_cov_matrix = sigma_sqr[it + 1] * np.linalg.inv(
    ((1/lambda_sqr[it]) * np.identity(X_cols) + np.dot(X.T, X)))
  sample = np.random.multivariate_normal(curr_mean_vector.flatten(), curr_cov_matrix, check_valid='warn')
  # try:
  #   sample = np.random.multivariate_normal(curr_mean_vector.flatten(), curr_cov_matrix, check_valid='raise')
  # except ValueError:
  #   print('Debug')
    
  return sample

def deltaSqrSampleSeqCoup(X, y, beta, mu, lambda_sqr, sigma_sqr, delta_sqr,
  X_cols, alpha_gamma_delta_sqr, beta_gamma_delta_sqr, it, change_points):
  # Get the posterior expectations of beta
  betasTilde = betaTildeSampler(y, X, mu, change_points, lambda_sqr[it], delta_sqr[it])

  accum = 0 # define the product accumulator
  for idx, cp in enumerate(change_points):
    if idx > 0: # the sum will take a value
      mu = betasTilde[idx - 1] # Get the posterior of the last segment
      currBeta = beta[it + 1][idx] # Get the betas vector from the segment
      X_cols_h = X_cols[idx] # Get the current cols of the current cp 
      el1 = np.dot(
        (currBeta - mu.flatten()).reshape(X_cols_h, 1).T,
        (currBeta - mu.flatten()).reshape(X_cols_h, 1)
      )
      accum += el1
  
  el2 = ((1/2) * (1 / sigma_sqr[it + 1]))
  betaMuSum = el2 * accum 
  H = len(change_points)

  # Calculate the parameters of the gamma
  a_gamma = alpha_gamma_delta_sqr + (H - 1) * (X_cols[0] / 2) 
  b_gamma = beta_gamma_delta_sqr + betaMuSum
  # Sample from the dist
  sample = 1 / (np.random.gamma(a_gamma, scale= (1 / b_gamma)))

  return sample

def lambdaSqrSamplerWithChangepointsSeqCoup(beta, sigma_sqr, X_cols, 
  alpha_gamma_lambda_sqr, beta_gamma_sigma_sqr, it, change_points):
  
  X_chols_h = X_cols[0] # columns for the first segment
  currBeta = beta[it + 1][0] # betas vector for the first segment
  currSigma = 1 / sigma_sqr[it + 1] # get the current sigma sqr value
  #H = len(change_points) # TODO change with marcos answer
  el = np.dot(currBeta.T, currBeta)
  a_gamma = alpha_gamma_lambda_sqr + ((len(beta[it][0]) + 1) / 2)
  b_gamma = beta_gamma_sigma_sqr + ((1/2) * currSigma * el)

  # Sample from the dist
  sample = 1 / (np.random.gamma(a_gamma, scale= (1/ b_gamma)))
  
  return sample # return the sampled value

  # sample from the proposed dist
def lambdaSqrSamplerWithChangepoints(X, beta, mu, sigma_sqr, X_cols,
  alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr, it, change_points):
  ################ 3(b) Get a sample of lambda square from a Gamma distribution
  # Get the current beta that was sampled from the changepoint
  accum = 0
  for idx, cp in enumerate(change_points):
    currBeta = beta[it + 1][idx] # Get the betas vector from the segment
    X_cols_h = X_cols[idx] # Get the current cols of the current cp 
    el1 = np.dot((currBeta - mu.flatten()).reshape(X_cols_h, 1).T, (currBeta - mu.flatten()).reshape(X_cols_h, 1))
    accum += el1
  
  el2 = ((1/2) * (1 / sigma_sqr[it + 1]))
  betaMuSum = el2 * accum 
  H = len(change_points) 

  # Calculate the parameters of the gamma
  a_gamma = alpha_gamma_lambda_sqr + H * ((len(beta[it][0]))/2) #TODO not hardcode the beta[0] Could change dims?
  b_gamma = beta_gamma_lambda_sqr + betaMuSum
  # Sample from the dist
  invSample = np.random.gamma(a_gamma, scale= (1/ b_gamma))
  sample = 1/(invSample)
  
  return sample

def lambdaSqrSampler(X, beta, mu, sigma_sqr, X_cols, alpha_gamma_lambda_sqr,
  beta_gamma_lambda_sqr, it):
  ################ 3(a) Get a sample of lambda square from a Gamma distribution
  el1 = np.dot((beta[it + 1] - mu.flatten()).reshape(X_cols,1).T, (beta[it + 1] - mu.flatten()).reshape(X_cols,1))  
  el2 = ((1/2) * (1 / sigma_sqr[it + 1]))
  a_gamma = alpha_gamma_lambda_sqr + ((X.shape[1])/2)
  b_gamma = beta_gamma_lambda_sqr + el2 * el1
  sample = 1/(np.random.gamma(a_gamma, scale= (1/ b_gamma)))
  
  return sample

def muSampler(mu, change_points, X, y, sigma_sqr, lambda_sqr):
  '''
    TODO documentation
  '''
  # construct the covar matrix
  cov = np.eye(mu.shape[0])
  # 1 -> Calculate the the sigma dagger dagger matrix
  acc = 0
  for idx, cp in enumerate(change_points):
    X_h = X[idx] # get the current X_h
    
    el1 = (sigma_sqr * np.eye(X_h.shape[0])) + sigma_sqr * lambda_sqr * np.dot(X_h, X_h.T)
    el1 = np.linalg.inv(el1)

    el2 = np.dot(np.dot(X_h.T, el1), X_h)
    acc = el2 + acc

  sigmaDaggerDagger = np.linalg.inv(acc + cov) # in this case we dont need to invert cov since its eye
  
  acc = 0
  for idx, cp in enumerate(change_points):
    X_h = X[idx]
    y_h = y[idx]

    el1 = (sigma_sqr * np.eye(X_h.shape[0])) + sigma_sqr * lambda_sqr * np.dot(X_h, X_h.T)
    el1 = np.linalg.inv(el1)
    el2 = np.dot(np.dot(X_h.T, el1), y_h.reshape(y_h.shape[0],1))
    acc = el2 + acc

  el3 = acc + np.dot(cov, mu) #TODO check mu
  muDaggerDagger = np.dot(sigmaDaggerDagger, el3)
  # Get the sample
  sample = np.random.multivariate_normal(muDaggerDagger.flatten(), sigmaDaggerDagger)
  # Calculate the density
  density = multivariate_normal.pdf(sample.flatten(), mean = muDaggerDagger.flatten(),
    cov = sigmaDaggerDagger)
  # densityBefore  = multivariate_normal.pdf(mu.flatten(), mean = muDaggerDagger.flatten(),
  #   cov = sigmaDaggerDagger)

  # reshape mu to avoid trouble later
  sample = sample.reshape(sample.shape[0], 1)
  
  #return sample, density, densityBefore
  return sample, density
