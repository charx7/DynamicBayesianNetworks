import numpy as np
from scipy.spatial.distance import cdist, squareform

# plooter imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def kernel(X1, X2, l=1.0, sigma_f=1.0):
  '''
    Square exponential kernel, computes the covariance matrix
    from points X.
    
    Note: when we use l for all input dimensions then the kernel
    becomes an (isotropic) kernel.

    Args:
      X1: first np array of points
      X2: second np array of points
      l: controls the smoothness of the kernel -> similar to the variance
      on a gaussian
      sigma_f: controls de vertical variation
    
    Returns:
      cov_matrix: covariance matrix 
  '''
  pairwise_sq_dist = cdist(X1, X2, 'sqeuclidean')
  exponential = np.exp((-0.5 / (l ** 2)) * pairwise_sq_dist)
  kernel = (sigma_f ** 2) * exponential 
  
  return kernel

def posterior_predictive(X_new, X_obs, y_obs, smoothness = 1.0, vert_variation = 1.0, resp_noise=1e-8):
  '''
    Computes the mean and variance estimators of the mean and variance of the posterior
    predictive distribution of the GP.
    
    Args:
      X_new: new values whose response is unknowkn
      X_obs: observed data points
      y_obs: observed response values
      smoothness: smoothness of the kernel function
      vert_variation: vertical variation of the kernel function
      resp_noise: associated response noise parameter
    
    Returns:
      mu_posterior: mean vector posterior estimator gp
      sigma_m_posterior: covariance matrix posterior estimator of the gp
  '''
  # calculate the kernel for observed values of X and new values of X
  obs_covar_m = kernel(X_obs, X_obs)
  inv_obs_covar_m = np.linalg.inv(obs_covar_m)
  K_star = kernel(X_obs, X_new, l = smoothness, sigma_f = vert_variation)
  K_star_star = kernel(X_new, X_new)
  K_y = obs_covar_m + (resp_noise ** 2) * np.eye(X_obs.shape[0])

  # estimator for the posterior mean vector
  mu_posterior = np.dot(np.dot(K_star.T, inv_obs_covar_m), y_obs)
  # estimator for the posterior sigma vector
  sigma_m_posterior = K_star_star - np.dot(np.dot(K_star.T, np.linalg.inv(K_y)), K_star)

  return mu_posterior, sigma_m_posterior

def generate_data():
  '''
    Generates 1d data points from -5 to 5 and 
    assigns y = sin(x) as a response (true function)
  '''
  # generate points from -5 to 5 with step 0.2
  X = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
  y = np.sin(X) # make the response the sin of X

  return X, y

def main():
  X_train, y_train = generate_data()

  X_new = np.arange(-5, 5, 0.2).reshape(-1, 1)
  mu_s, cov_s = posterior_predictive(X_new, X_train, y_train)

  # get the samples from the extracted mean and cov estimated posterior vectors
  samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
  
  # plot the gp
  plot_gp(mu_s, cov_s, X_new, X_train=X_train, Y_train=y_train, samples=samples)

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show() 

if __name__ == '__main__':
  main()
