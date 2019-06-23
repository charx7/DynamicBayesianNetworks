import numpy as np

def sigmaSqrSampler(y, X, mu, lambda_sqr, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, numSamples, T, it):
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
  
  return curr_sigma_sqr

def betaSampler(y, X, mu, lambda_sqr, sigma_sqr, X_cols, numSamples, T, it):
  # Mean Vector Calculation
  el1 = np.linalg.inv(((1/(lambda_sqr[it])) * np.identity(X_cols)) + np.dot(X.T, X))
  el2 = ((1/(lambda_sqr[it])) * mu) + np.dot(X.T, y.reshape(numSamples, 1))
  curr_mean_vector = np.dot(el1, el2)
  # Sigma vector Calculation
  curr_cov_matrix = sigma_sqr[it + 1] * np.linalg.inv(((1/lambda_sqr[it]) * np.identity(X_cols) + np.dot(X.T, X)))
  sample = np.random.multivariate_normal(curr_mean_vector.flatten(), curr_cov_matrix)
  
  return sample

def lambdaSqrSampler(X, beta, mu, sigma_sqr, X_cols, alpha_gamma_lambda_sqr, beta_gamma_lambda_sqr, it):
  el1 = np.dot((beta[it + 1] - mu.flatten()).reshape(X_cols,1).T, (beta[it + 1] - mu.flatten()).reshape(X_cols,1))  
  el2 = ((1/2) * (1/sigma_sqr[1]))
  a_gamma = alpha_gamma_lambda_sqr + ((X.shape[1])/2)
  b_gamma = beta_gamma_lambda_sqr + el2 * el1
  sample = 1/(np.random.gamma(a_gamma, scale= (1/ b_gamma)))
  
  return sample

if __name__ == '__main__':
  print('Test')