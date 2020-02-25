import numpy as np
import pandas as pd
from utils import read_pd_dataframe

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from gaussian_process import GP

def fit_mTor():
  data = read_pd_dataframe('./data/mTOR_data.csv') 
  
  # select the treatment AS001_EQ
  treatment = 'AS001_EQ'
  norm_method = 'Normalized to loading control (GAPDH) for specific value at each time point'
  df = data[(data.name == treatment) & (data.method == norm_method)]
  
  keep_cols = [
    'aa + ins [min]',
    'IR-beta-pY1146',
    'IRS-pS636/639',
    'AMPK-pT172',
    'TSC2-pS1387',
    'Akt-pT308',
    'Akt-pS473',
    'mTOR-pS2448',
    'mTOR-pS2481',
    'p70-S6K-pT389',
    'PRAS40-pS183',
    'PRAS40-pT246'
    #'PRAS40-pT246.1' #-> current treatment doesnt have this column
  ]
  df = df[keep_cols] # subset the columns to keep
  df = df.rename(columns = {'aa + ins [min]': 'time'}) # rename the time column
  #df = df.iloc[:len(df) - 2] # drop the last idx -> impossible to fit a gp

  curr_column = 'IR-beta-pY1146' 
  X_train = df['time'].to_numpy().reshape(-1,1) # we need explicitely make it nx1 matrix
  y_train = df[curr_column].to_numpy().reshape(-1,1) # we need explicitely make it nx1 matrix
  y_train = y_train - np.mean(y_train) # center the data
  y_train = y_train / np.std(y_train) # scale

  noise = 1e-8
  gp = GP(X_train, y_train, noise=noise)
  gp.fit() # optimize
  X_new = np.arange(0, 125, 0.1).reshape(-1, 1) # to get the prediction curves

  mu_post, cov_post = gp.pred(X_new)
  samples = np.random.multivariate_normal(mu_post.ravel(), cov_post, 3) # get a posterior sample 3 times

  GP.plot_gp(mu_post, cov_post, X_new, X_train, y_train, samples = samples)

  # SK-learn
  # # parameters of the gp
  # noise = 1e-8 # we are not considering noise in our sampled data
  # rbf = ConstantKernel(1.0) * RBF(length_scale=1.0) # kernel of our gp
  # gpr = GaussianProcessRegressor(kernel = rbf, alpha = noise ** 2)

  # # fit the gp
  # gpr.fit(X_train, y_train)

  # # Compute posterior predictive mean and covariance
  # X_pred = np.arange(0, 125, 0.1).reshape(-1, 1) # sample every 1 min
  # mu_s, cov_s = gpr.predict(X_pred, return_cov=True)
  # samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3) # get a posterior sample 3 times

  # # Obtain optimized kernel parameters
  # l = gpr.kernel_.k2.get_params()['length_scale']
  # sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])
  
  # GP.plot_gp(mu_s, cov_s, X_pred, X_train = X_train, Y_train = y_train, samples = samples)

def fit_example_data():
  X_train, y_train = GP.generate_data()
  noise = 1e-8
  X_new = np.arange(-5, 5, 0.2).reshape(-1, 1)
  
  gp = GP(X_train, y_train, noise) # make a gp object
  gp.fit()
  mu_s, cov_s = gp.pred(X_new)

  # get the samples from the extracted mean and cov estimated posterior vectors
  samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
  
  # plot the gp
  GP.plot_gp(mu_s, cov_s, X_new, X_train=X_train, Y_train=y_train, samples=samples)

  ### sklearn to compare if the ll optimizations are correct
  # rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
  # gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)
  # # Reuse training data from previous 1D example
  # gpr.fit(X_train, y_train)

  # # Compute posterior predictive mean and covariance
  # mu_s, cov_s = gpr.predict(X_new, return_cov=True)

  # # Obtain optimized kernel parameters
  # l = gpr.kernel_.k2.get_params()['length_scale']
  # sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

  # GP.plot_gp(mu_s, cov_s, X_new, X_train=X_train, Y_train=y_train)

if __name__ == '__main__':
  fit_mTor()
