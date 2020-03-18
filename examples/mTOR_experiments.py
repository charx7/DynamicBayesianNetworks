import numpy as np
import pandas as pd
from utils import read_pd_dataframe

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from gaussian_process import GP

def query_data(treatment, insulin, norm_method, protein):
  data = read_pd_dataframe('./data/mTOR_data.csv') 
  
  if treatment == 'AVERAGE':
    treatments_list = ['AS001_EQ', 'AS001_EV', 'AS001_FM']
    df = data[(data.name.isin(treatments_list)) & (data.method == norm_method) & (data.insulin == insulin)]
    
    proteins = [
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

    series_vec = []
    for treatment in treatments_list:
      series_list = [df[protein][data.name == treatment] for protein in proteins]
      series_vec.append(series_list)
    
    avg_df = pd.DataFrame() # empty df
    for jdx in range(len(series_list)):
      avg_series = pd.Series([])
      for idx in range(len(series_vec)):
        curr_series = series_vec[idx][jdx]
        if avg_series.empty:
          avg_series = curr_series.reset_index(drop = True)
        else:
          # if we have empty values in the series we impute the col mean
          if np.sum(curr_series.isna()):
            col_mean = curr_series.mean(skipna = True)
            curr_series = curr_series.fillna(col_mean)

          avg_series = avg_series.add(curr_series.reset_index(drop = True)) # sum the current protein

      avg_series = avg_series.divide(len(series_vec)) # divide by the total number of proteins
      curr_protein = proteins[jdx] + '_AVERAGE'
      avg_series = avg_series.rename(curr_protein)
      #avg_series = avg_series.to_frame()
      avg_df[curr_protein] = avg_series
      #avg_df = avg_df.append(avg_series, ignore_index = True)

    X_train = df['aa + ins [min]'].iloc[0:10].to_numpy().reshape(-1,1) # we need explicitely make it nx1 matrix
    y_train = avg_df[protein + '_AVERAGE'].to_numpy().reshape(-1,1) # we need explicitely make it nx1 matrix

  else: # select the treatment norm-method and insulin 
    df = data[(data.name == treatment) & (data.method == norm_method) & (data.insulin == insulin)]
  
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

    X_train = df['time'].to_numpy().reshape(-1,1) # we need explicitely make it nx1 matrix
    y_train = df[protein].to_numpy().reshape(-1,1) # we need explicitely make it nx1 matrix

  # if we find missing values in the col we impute the mean
  if np.sum(np.isnan(y_train)):
    no_missing = [x for x in y_train if np.isnan(x) == False]
    col_mean = np.mean(no_missing)
    inds = np.where(np.isnan(y_train))
    y_train[inds] = np.take(col_mean, inds[1])  

  y_train = y_train - np.mean(y_train) # center the data
  y_train = y_train / np.std(y_train) # scale

  return X_train, y_train

def fit_mTor_example():
  treatment = 'AS001_EQ'
  insulin = 'yes'
  protein = 'mTOR-pS2481' 
  norm_method = 'Normalized to loading control (GAPDH) for specific value at each time point'
  
  X_train, y_train = query_data(treatment, insulin, norm_method, protein)
  noise = 1e-8
  gp = GP(X_train, y_train, noise=noise)
  gp.fit() # optimize
  X_new = np.arange(0, 125, 0.1).reshape(-1, 1) # to get the prediction curves

  mu_post, cov_post = gp.pred(X_new)
  samples = np.random.multivariate_normal(mu_post.ravel(), cov_post, 3) # get a posterior sample 3 times

  plt.title(f'treatment = {treatment}, protein = {protein}, insulin = {insulin}')
  GP.plot_gp(mu_post, cov_post, X_new, X_train, y_train, samples = samples)

def mTOR_generate_data():
  treatment = 'AVERAGE'
  insulin = 'yes'
  norm_method = 'Normalized to loading control (GAPDH) for specific value at each time point'
  proteins = [
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

  gp_matrix_fit = np.array([]) 
  for protein in proteins:
    X_train, y_train = query_data(treatment, insulin, norm_method, protein)
    noise = 1e-8

    # LOOCV - fit for the current protein
    cv_opt_mse = float('inf')
    cv_opt_smoothness = None
    cv_opt_vert_variation = None
    gp_opt = None
    mse_vector = []
    loo = LeaveOneOut()
    loo.get_n_splits(X_train)
    for train_index, test_index in loo.split(X_train):
      curr_X_train, curr_X_test = X_train[train_index], X_train[test_index]
      curr_y_train, curr_y_test = y_train[train_index], y_train[test_index]
      # fit the current training data
      gp = GP(curr_X_train, curr_y_train, noise=noise)
      try:
        gp.fit() # optimize
        # current hyper-params
        curr_mle_smoothness = gp.smoothness_opt
        curr_mle_vert_variation = gp.vert_variation_opt
        # get the prediction mean + cov value and calculate mse
        mu_post, _ = gp.pred(curr_X_test)
        curr_mse = (curr_y_test - mu_post) ** 2
        mse_vector.append(curr_mse)
        if curr_mse < cv_opt_mse:
          gp_opt = gp
          cv_opt_mse = curr_mse
          cv_opt_smoothness = curr_mle_smoothness
          cv_opt_vert_variation = curr_mle_vert_variation
      except:
        pass # singular matrix -> go to the next iteration of loocv

    # select the best model and then run predictions
    X_new = np.arange(0, 120, 1).reshape(-1, 1) # to get the prediction curves
    # the mu_post will be our equidistant time-points
    mu_post, cov_post = gp_opt.pred(X_new)
    
    # stack the current mu_post to a numpy matrix
    gp_matrix_fit = np.concatenate((gp_matrix_fit, mu_post), axis = 1) if gp_matrix_fit.size else mu_post

  # transform the numpy matrix into a pandas df
  fitted_df = pd.DataFrame(data = gp_matrix_fit, columns = proteins)
  path = './examples/data/gp_mTOR.csv' 
  fitted_df.to_csv(path, index = False)

def loocv_fit_mTor():
  treatment = 'AVERAGE'
  insulin = 'yes'
  norm_method = 'Normalized to loading control (GAPDH) for specific value at each time point'
  protein = 'PRAS40-pT246' 

  X_train, y_train = query_data(treatment, insulin, norm_method, protein)
  noise = 1e-8

  # LOOCV - example
  cv_opt_mse = float('inf')
  cv_opt_smoothness = None
  cv_opt_vert_variation = None
  gp_opt = None
  mse_vector = []
  loo = LeaveOneOut()
  loo.get_n_splits(X_train)
  for train_index, test_index in loo.split(X_train):
    curr_X_train, curr_X_test = X_train[train_index], X_train[test_index]
    curr_y_train, curr_y_test = y_train[train_index], y_train[test_index]
    # fit the current training data
    gp = GP(curr_X_train, curr_y_train, noise=noise)
    try:
      gp.fit() # optimize
      # current hyper-params
      curr_mle_smoothness = gp.smoothness_opt
      curr_mle_vert_variation = gp.vert_variation_opt
      # get the prediction mean + cov value and calculate mse
      mu_post, _ = gp.pred(curr_X_test)
      curr_mse = (curr_y_test - mu_post) ** 2
      mse_vector.append(curr_mse)
      if curr_mse < cv_opt_mse:
        gp_opt = gp
        cv_opt_mse = curr_mse
        cv_opt_smoothness = curr_mle_smoothness
        cv_opt_vert_variation = curr_mle_vert_variation
    except:
      pass # singular matrix -> go to the next iteration of loocv

  # select the best model and then run predictions
  X_new = np.arange(0, 125, 0.1).reshape(-1, 1) # to get the prediction curves

  mu_post, cov_post = gp_opt.pred(X_new)
  samples = np.random.multivariate_normal(mu_post.ravel(), cov_post, 3) # get a posterior sample 3 times
  
  plt.title(f'treatment = {treatment}, protein = {protein}, insulin = {insulin} \n l = {cv_opt_smoothness} sigma_f = {cv_opt_vert_variation}')
  GP.plot_gp(mu_post, cov_post, X_new, X_train, y_train, samples = samples)
  print('The cv mse was: ', np.mean(mse_vector))

def fit_example_data():
  X_train, y_train = GP.generate_data(0.6)
  noise = 1e-8
  X_new = np.arange(-5, 5, 0.1).reshape(-1, 1)
  
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

def noisy_data_example():
  X_train, y_train = GP.generate_data(0.2)

  noise = 0.2
  X_new = np.arange(-4.5, 4, 0.1).reshape(-1, 1)
  gp = GP(X_train, y_train, noise) # make a gp object
  gp.fit()

  #gp._smoothness = 1
  #gp._vert_variation = 3
  mu_s, cov_s = gp.pred(X_new)

  # get the samples from the extracted mean and cov estimated posterior vectors
  samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
  
  # plot the gp
  plt.title(f'l = {gp.smoothness_opt}, sigma_f = {gp.vert_variation_opt}, sigma_y = {gp._noise}')
  GP.plot_gp(mu_s, cov_s, X_new, X_train=X_train, Y_train=y_train, samples = samples)

def deterministic_data_example():
  X_train, y_train = GP.generate_data(1e-8)

  noise = 1e-8
  X_new = np.arange(-4.5, 4, 0.1).reshape(-1, 1)
  gp = GP(X_train, y_train, noise) # make a gp object
  
  gp._smoothness = 1
  gp._vert_variation = 1
  mu_s, cov_s = gp.pred(X_new)

  # get the samples from the extracted mean and cov estimated posterior vectors
  samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 6)
  
  # plot the gp
  plt.title(f'l = {gp._smoothness}, sigma_f = {gp._vert_variation}, sigma_y = {gp._noise}')
  GP.plot_gp(mu_s, cov_s, X_new, X_train=X_train, Y_train=y_train, samples=samples)

if __name__ == '__main__':
  #loocv_fit_mTor()
  mTOR_generate_data()