from scipy.special import gamma
import numpy as np
import math 

def calculateMarginalLikelihood(X, y, mu, alpha_sigma, beta_sigma, lambda_sqr, num_samples):
  print('Calculating...')
  T = num_samples
  cMatrix = np.identity(num_samples) + lambda_sqr * (np.dot(X, X.T))
  cMatrixDeterminant = np.linalg.det(cMatrix)

  el1 = gamma(T/2 + alpha_sigma) / gamma(alpha_sigma)
  el2 =  (((math.pi) ** (-T/2)) * (2 * beta_sigma) ** (alpha_sigma)) / (cMatrixDeterminant ** (1/2))
  el3 = (2 * beta_sigma + np.dot(np.dot((y.reshape(num_samples, 1) - np.dot(X, mu)).T , np.linalg.inv(cMatrix)), (y.reshape(num_samples, 1) - np.dot(X, mu)))) ** (T/2 + alpha_sigma) 

  res = el1 * el2 * el3
  return res

def testFeaturesPriorProb():
  print('Executing test for calculation of marginal likelihood probs for a feature set...')
  dummyData = np.array([1, 3, 6])
  calculateMarginalLikelihood()

if __name__ == '__main__':
  testFeaturesPriorProb()
  