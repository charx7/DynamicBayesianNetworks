from scipy.special import gamma
import numpy as np
import math 

def calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_sigma,
  beta_sigma, lambda_sqr, num_samples, change_points):
  
  T = num_samples
  # Now the C Matrix needs to be calculated as a part of a product
  accumProd = 1 # Initialize the accumulator
  cMatrixVector = []
  for idx, cp in enumerate(change_points):
    currCplen = y[idx].shape[0] # The current cp len
    X_h = X[idx] # Get the current design matrix (inside a cp)
    
    cMatrix = np.identity(currCplen) + lambda_sqr * (np.dot(X_h, X_h.T))
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

  el3 = (2 * beta_sigma + accumSum) ** (-T/2 + alpha_sigma)
  res = el1 * el2 * el3 # Calculate the final result

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
  el3 = (2 * beta_sigma + np.dot(np.dot(matrixElement1, matrixElement2), matrixElement3)) ** (-T/2 + alpha_sigma) 

  res = el1 * el2 * el3

  return res

def testFeaturesPriorProb():
  print('Executing test for calculation of marginal likelihood probs for a feature set...')
  dummyData = np.array([1, 3, 6])
  calculateMarginalLikelihood()

if __name__ == '__main__':
  testFeaturesPriorProb()
  