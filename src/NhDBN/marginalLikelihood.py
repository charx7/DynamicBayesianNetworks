from scipy.special import gamma
import numpy as np
import math 

def calculateSeqCoupMargLikelihoodWithChangepoints(X, y, mu, alpha_sigma,
  beta_sigma, lambda_sqr, delta_sqr, num_samples, change_points):
  '''
    Missing Documentation for this function
  '''
  T = num_samples
  # Now the C Matrix needs to be calculated as a part of a product
  accumProd = 1 # Initialize the accumulator
  cMatrixVector = []
  for idx, cp in enumerate(change_points):
    currCplen = y[idx].shape[0] # The current cp len
    X_h = X[idx] # Get the current design matrix (inside a cp)
    
    if idx == 0: # we are on the first changepoint
      cMatrix = np.identity(currCplen) + lambda_sqr * (np.dot(X_h, X_h.T))
    else: # we are on the other changepoints
      cMatrix = np.identity(currCplen) + delta_sqr * (np.dot(X_h, X_h.T))

    cMatrixVector.append(cMatrix) # append the C matrix because we will use it later
    cMatrixDeterminant = np.linalg.det(cMatrix) # Determinant of the curr C matrix
    cMatrixDeterminantSqrt = cMatrixDeterminant ** (1/2)

    accumProd = accumProd * cMatrixDeterminantSqrt # Acculate the product
  
  el1 = gamma(T/2 + alpha_sigma) / gamma(alpha_sigma)
  el2 =  (((math.pi) ** (-T/2)) * (2 * beta_sigma) ** (alpha_sigma)) / (accumProd) # Now we need the accum
  
  accumSum = 0
  for idx, cp in enumerate(change_points):
    currCplen = y[idx].shape[0]
    y_h = y[idx] # Get the current sub y vector
    X_h = X[idx] # Get the current design matrix (inside a cp)
    # TODO do the same thing with the \mu vector
    cMatrix_h = cMatrixVector[idx] # Get the change point C Matrix
    # Matrix multiplication elements
    matrixElement1 = (y_h.reshape(currCplen, 1) - np.dot(X_h, mu)).T
    matrixElement2 = np.linalg.inv(cMatrix_h)
    matrixElement3 = y_h.reshape(currCplen, 1) - np.dot(X_h, mu)
    partial = np.dot(np.dot(matrixElement1, matrixElement2), matrixElement3)
    accumSum = accumSum + partial

  el3 = (2 * beta_sigma + accumSum) ** -(T/2 + alpha_sigma)
  res = el1 * el2 * el3 # Calculate the final result

  return res

def calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_sigma,
  beta_sigma, lambda_sqr, num_samples, change_points, method='', delta_sqr=[]):
  '''
    Missing Documentation for this function
  '''
  T = num_samples
  # Now the C Matrix needs to be calculated as a part of a product
  accumProd = 0 # Initialize the accumulator
  cMatrixVector = []
  for idx, cp in enumerate(change_points):
    currCplen = y[idx].shape[0] # The current cp len
    X_h = X[idx] # Get the current design matrix (inside a cp)
    
    if idx > 0 and method == 'seq-coup': # we are on the first changepoint
      cMatrix = np.identity(currCplen) + delta_sqr * (np.dot(X_h, X_h.T))
    else: # we are on the other changepoints
      cMatrix = np.identity(currCplen) + lambda_sqr * (np.dot(X_h, X_h.T))

    cMatrixVector.append(cMatrix) # append the C matrix because we will use it later
    cMatrixDeterminant = np.linalg.det(cMatrix) # Determinant of the curr C matrix
    #cMatrixDeterminantSqrt = cMatrixDeterminant ** (1/2)
    logcMatrixDeterminant = math.log(cMatrixDeterminant ** (1/2))

    accumProd = accumProd + logcMatrixDeterminant# Acculate the log
  
  el1 = math.log(gamma(T/2 + alpha_sigma)) - math.log(gamma(alpha_sigma))
  #el2 =  (((math.pi) ** (-T/2)) * (2 * beta_sigma) ** (alpha_sigma)) / (accumProd) # Now we need the accum
  el2 = math.log(math.pi ** (-T/2)) + math.log((2 * beta_sigma) ** (alpha_sigma)) - accumProd

  accumSum = 0
  for idx, cp in enumerate(change_points):
    if method == 'seq-coup': # if its seq coup then get the beta tilde
      betaTilde = mu[idx]
    else:
      betaTilde = mu # if not mu is just the zero vector

    currCplen = y[idx].shape[0]
    y_h = y[idx] # Get the current sub y vector
    X_h = X[idx] # Get the current design matrix (inside a cp)
    
    cMatrix_h = cMatrixVector[idx] # Get the change point C Matrix
    # Matrix multiplication elements
    matrixElement1 = (y_h.reshape(currCplen, 1) - np.dot(X_h, betaTilde)).T
    #matrixElement2 = np.linalg.inv(cMatrix_h)
    # #### debug
    try: 
      matrixElement2 = np.linalg.inv(cMatrix_h)
    except np.linalg.LinAlgError as err:
      if 'Singular matrix' in str(err):
        print('investigate')
    ####
    matrixElement3 = y_h.reshape(currCplen, 1) - np.dot(X_h, betaTilde)
    partial = np.dot(np.dot(matrixElement1, matrixElement2), matrixElement3)
    accumSum = accumSum + partial

  el3 = -(T/2 + alpha_sigma) * math.log((2 * beta_sigma + accumSum)) 

  res = el1 + el2 + el3 # Calculate the final result

  return res

def calculateMarginalLikelihood(X, y, mu, alpha_sigma, beta_sigma, lambda_sqr, num_samples):
  T = num_samples
  cMatrix = np.identity(num_samples) + lambda_sqr * (np.dot(X, X.T))
  cMatrixDeterminant = np.linalg.det(cMatrix)

  el1 = gamma(T/2 + alpha_sigma) / gamma(alpha_sigma)
  el2 =  (((math.pi) ** (-T/2)) * (2 * beta_sigma) ** (alpha_sigma)) / (cMatrixDeterminant ** (1/2))
  matrixElement1 = (y.reshape(num_samples, 1) - np.dot(X, mu)).T
  matrixElement2 = np.linalg.inv(cMatrix)
  matrixElement3 = y.reshape(num_samples, 1) - np.dot(X, mu)
  el3 = (2 * beta_sigma + np.dot(np.dot(matrixElement1, matrixElement2), matrixElement3)) ** -(T/2 + alpha_sigma) 

  res = el1 * el2 * el3

  return res
  