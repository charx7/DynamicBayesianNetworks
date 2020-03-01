import numpy as np
from numpy.linalg import cholesky, det, lstsq
import pandas as pd
from scipy.spatial.distance import cdist, squareform
from scipy.optimize import minimize
from utils import read_pd_dataframe

# plooter imports
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class GP():
  '''
    Defines a Gaussian Process accoding to the characteristics
    
    Args:
      X_train: training locations (m x d).
      Y_train: training targets (m x 1).
      smoothness: smoothness of the kernel function
      vert_variation: vertical variation of the kernel function
      resp_noise: associated response noise parameter
  '''
  def __init__(self, X_train, y_train, noise = 1e-8):
    self._X_train = X_train
    self._y_train = y_train
    self._noise = noise
    self._smoothness = 1.0
    self._vert_variation = 1.0
    self.smoothness_opt = None
    self.vert_variation_opt = None
    self.nll_func = None

  def fit(self, naive = True):
    '''
      Fits the Gp according to the training data using a naive nll function
      -> this could be changed to use marcos approach fo the nll

      Args:
        naive: which nll func are we going to use 
    '''
    curr_nll = self.nll_fn(naive) # which nll function we will use
    
    # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
    # TODO run several times to avoid local minima
    min_nll = minimize(curr_nll, [1, 1], 
                  bounds=((1e-5, None), (1e-5, None)),
                  method='L-BFGS-B')

    # set the min values to the gp obj    
    self.smoothness_opt, self.vert_variation_opt = min_nll.x 

  def pred(self, X_new):
    '''
      Computes the mean and variance estimators of the mean and variance of the posterior
      predictive distribution of the GP.
      
      Args:
        X_new: new values whose response is unknowkn
        
      Returns:
        mu_posterior: mean vector posterior estimator gp
        sigma_m_posterior: covariance matrix posterior estimator of the gp
    '''
    # if the optimized parameters exists then set them up
    if (self.smoothness_opt != None and self.vert_variation_opt != None):
      smoothness = self.smoothness_opt
      vert_variation = self.vert_variation_opt
    else:
      smoothness = self._smoothness
      vert_variation = self._vert_variation

    # calculate the kernel for observed values of X and new values of X
    obs_covar_m = GP.kernel(self._X_train, self._X_train, l = smoothness, sigma_f = vert_variation)
    inv_obs_covar_m = np.linalg.inv(obs_covar_m)
    K_star = GP.kernel(self._X_train, X_new, l = smoothness , sigma_f = vert_variation)
    K_star_star = GP.kernel(X_new, X_new, l = smoothness, sigma_f = vert_variation)
    K_y = obs_covar_m + (self._noise ** 2) * np.eye(self._X_train.shape[0])

    # estimator for the posterior mean vector
    mu_posterior = np.dot(np.dot(K_star.T, inv_obs_covar_m), self._y_train)
    # estimator for the posterior sigma vector
    sigma_m_posterior = K_star_star - np.dot(np.dot(K_star.T, np.linalg.inv(K_y)), K_star)

    return mu_posterior, sigma_m_posterior

  def nll_fn(self, naive = True):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given 
    noise level.
    
    Args:
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of the neg LL, if 
               False use a numerically more stable implementation. 
        
    Returns:
        Minimization objective.
    '''
    def nll_naive(theta):
        # Naive implementation of the NegLL Works well for the examples 
        # in this article but is numerically less stable compared to 
        # the implementation in nll_stable below.
        K = self.kernel(self._X_train, self._X_train, l = theta[0], sigma_f=theta[1]) + \
            self._noise ** 2 * np.eye(len(self._X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * self._y_train.T.dot(np.linalg.inv(K).dot(self._y_train)) + \
               0.5 * len(self._X_train) * np.log(2 * np.pi)

    def nll_stable(theta):
        # Numerically more stable implementation of the LL using cholesky decomposition
        K = self.kernel(self._X_train, self._X_train, l = theta[0], sigma_f = theta[1]) + \
            self._noise ** 2 * np.eye(len(self._X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * self._y_train.T.dot(lstsq(L.T, lstsq(L, self._y_train)[0])[0]) + \
               0.5 * len(self._X_train) * np.log(2 * np.pi)
    
    if naive:
        return nll_naive
    else:
        return nll_stable

  @staticmethod
  def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    '''
      Static method that plots our data with alongside 95% confidence
      interval and sample realization from the mvg if given
    '''
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    uncertainty = np.nan_to_num(uncertainty) # replace the possible 0s
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show() 

  @staticmethod
  def generate_data(noise):
    '''
      Generates 1d data points from -5 to 5 and 
      assigns y = sin(x) as a response (true function)
    '''
    # generate points from -5 to 5 with step 0.2
    X = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
    y = np.sin(X)
    # np.random.seed(42) # set seed
    # noise = noise
    # X = np.arange(-3, 4, 1).reshape(-1, 1)
    # y = np.sin(X) + noise * np.random.randn(*X.shape) # make the response the sin of X
    
    return X, y

  @staticmethod
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
