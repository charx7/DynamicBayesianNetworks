import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from utils import load_chain

def get_feats_resp(network_config):
  # get the response according to the network config
  responses_list = list(network_config['features'].keys())
  col_features_set = set([string.lstrip('X') for string in responses_list])
  total_feats_set = set([str(idx) for idx in range(len(col_features_set) + 1)])
  curr_response = (total_feats_set - col_features_set).pop() # popping is the only way to retreive an element from a set
  total_feats_list = [int(string.lstrip('X')) for string in responses_list]

  return total_feats_list, curr_response

def boxplot(data, feature, response):
  #### With matplotlib
  # # Create a figure instance
  # fig = plt.figure(1, figsize=(9, 6))

  # # Create an axes instance
  # ax = fig.add_subplot(111)

  # # Create the boxplot
  # box_plot = ax.boxplot(data)

  # plt.show() # show
  
  #### With seaborn
  title = 'Edge ' + str(feature) + ' -> ' + response + ' Boxplot Betas Values Across Time'
  col_names = [i for i in range(data.shape[1] + 1)]
  col_names = [i for i in col_names if i > 0] 
  sns.set(style="darkgrid") # style
  df = pd.DataFrame(data, columns= col_names)
  ax = sns.boxplot(data=df).set(xlabel='time-points', ylabel='beta-value', title = title)
 
  # Add jitter with the swarmplot function.
  #ax = sns.swarmplot(data=df, color="grey") # seems too clunky
  
  figure_route = 'figures/edge_'+ str(feature) + '_' + response + '_boxplot_betas_overtime'
  plt.savefig(figure_route)
  plt.clf() # clear the figure obj from memory so it doesnt overlap
  #plt.show() #TODO make it as an agument

def cps_plot(data, resp_label):
  _x = [i for i in range(33)]
  _x = [i for i in _x if i > 1]

  title = 'Cps Probability Across Time of the Chain Configuration Y = X' + resp_label
  sns.set(style="darkgrid") # style
  ax = sns.barplot(
    y = data,
    x = _x,
    palette="Blues_d",
    label="cps-prob-over-time").set(xlabel='time-points', ylabel='cps-prob')
  #plt.xlim(2, None) # show from 2 in case we want a line-plot
  plt.title(title)
  figure_route = 'figures/changepoints_prob_X' + resp_label
  plt.savefig(figure_route)
  plt.clf() # clear figure
  #plt.show()

def line_plot(data, parent, response):
  title = 'Edge Fraction-Score of ' + parent + '->' + response 
  
  _x = [i for i in range(34)]
  _x = [i for i in _x if i > 0]

  ax = sns.lineplot(
    x=_x,
    y=data,
    markers=True,
    dashes=False)
  plt.title(title)
  plt.ylim(-0.55, 0.55)
  figure_route = 'figures/edge_fraction_' + parent + '_' + response
  plt.savefig(figure_route)
  plt.clf() # clear the figure so plots wont overlap
  

def fraction_scores_plot(network_configurations, scores_over_time_list):
  for idx, network_config in enumerate(network_configurations):
    feature_labels, response_label = get_feats_resp(network_config) 
    
    curr_scores_over_time = scores_over_time_list[idx] # get the current scores over time
    
    # TODO we need to add the intercept in this plots
    for j_idx, feature in enumerate(feature_labels): # loop over all features
      curr_edge = curr_scores_over_time[:,j_idx] 
      line_plot(curr_edge, str(feature), response_label)

def get_design_matrices_list(configurations):
  design_list = []

  data_dims = len(list(configurations[0]['features'].keys())) + 1 # get the dims of the data
  features_idx = [el for el in range(data_dims)] # list of possible idx

  for conf in configurations:
    X_dict = conf['features'] # get the current config dict

    # build the design matrix for the curr network config
    curr_design_matrix = np.ones([33, 1]) # ones vector    
  
    for idx in features_idx:
      curr_key = 'X' + str(idx) # make the cant key to get from the dict
      try:
        col_design_vector = X_dict[curr_key] # get the vector for dim idx
        col_design_vector = col_design_vector.reshape(-1, 1) # reshape for stacking
        curr_design_matrix = np.hstack((curr_design_matrix, col_design_vector))
      except KeyError:
        pass
    # append to the design matrices list
    design_list.append(curr_design_matrix)
  
  return design_list

def get_response_list(network_configs):
  resp_list = [] # empty resp list
  for config in network_configs:
    # append the response vector for each config
    resp_list.append(config['response']['y'])

  return resp_list

def boxplot_res(data, response):
  title = 'Residuals over time of response Y = X' + response
  col_names = [i for i in range(data.shape[1] + 1)]
  col_names = [i for i in col_names if i > 0] 
  sns.set(style="darkgrid") # style
  df = pd.DataFrame(data, columns= col_names)
  ax = sns.boxplot(data=df).set(xlabel='time-points', ylabel='residual-value', title = title)
 
  # Add jitter with the swarmplot function.
  #ax = sns.swarmplot(data=df, color="grey") # seems too clunky
  figure_route = 'figures/residuals_over_time_X' + response
  plt.savefig(figure_route)
  plt.clf() # clear fig
  #plt.show()


def residual_plots_overtime(betas_response_list, network_configs):
  m_list = get_design_matrices_list(network_configs) # get the list of design matrices
  r_list = get_response_list(network_configs) # get the list of response matrices
  
  for idx, matrix_config  in enumerate(m_list):
    # get the labels of the features(covariates) and response
    _, curr_response = get_feats_resp(network_configs[idx])
    # get the betas for the current config
    curr_response_betas = betas_response_list[idx]
    # get the responses for the first config
    curr_response_list = r_list[idx]

    curr_residuals_matrix = np.array([])
    # get the betas for every time point
    for time, curr_time_pt_betas in enumerate(curr_response_betas):
      curr_x = matrix_config[time] # get the x of the curr time point
      curr_x = curr_x.reshape(-1,1) # reshape + transpose 

      preds = np.dot(curr_time_pt_betas, curr_x) # compute pred values
      resp = curr_response_list[time] # get the current resp

      residuals = resp - preds # compute the residuals vector
      
      # horizontally stack them into a matrix
      curr_residuals_matrix = np.hstack((curr_residuals_matrix, residuals)) if curr_residuals_matrix.size else residuals

    # plot it!
    boxplot_res(curr_residuals_matrix, curr_response)

def beta_boxplots_overtime(betas_response_list, network_configurations):
  for jdx, network_config in enumerate(network_configurations):
    # get the betas for the current response configuration
    curr_response_betas = betas_response_list[jdx]

    total_feats_list, curr_response = get_feats_resp(network_config)

    # loop for every element in the features
    for idx, feat in enumerate(total_feats_list):
      curr_response_matrix = np.array([])
      # get the betas for every time point
      for curr_time_pt_betas in curr_response_betas:
        # betas for the current edge
        curr_edge_betas = curr_time_pt_betas[:,idx + 1] # select the spcific edge column +1 bc the intercept in 0th position
        curr_edge_betas = curr_edge_betas.reshape(curr_edge_betas.shape[0] , 1) # reshape for the horizontal stack
        # horizontally stack them into a matrix
        curr_response_matrix = np.hstack((curr_response_matrix, curr_edge_betas)) if curr_response_matrix.size else curr_edge_betas

      # plot it!
      boxplot(curr_response_matrix, feat, curr_response)

def chain_points_prob_plot(cps_over_response, network_configs):
  for idx, network_config in enumerate(network_configs):
    _, curr_resp_label = get_feats_resp(network_config) # get the labels for the plot/file

    network_config = network_config['response']['y'] # get the numpy array values
    #### Now the chainpoints prob vector
    thinned_changepoints = cps_over_response[idx] 

    # get the vector of each cps probability
    cps_prob = []
    time_periods = len(network_config)
    total_cps = np.sum([1 for cps in thinned_changepoints])
    for idx in range(time_periods):
      if (idx >1): # we dont start with the 0th or 1st element
        # check if idx in the cps vector
        counts_vector = [1 for cps in thinned_changepoints if idx in cps]
        counts = np.sum(counts_vector)
        fraq = counts / total_cps
        if counts != 0:
          cps_prob.append(fraq)
        else:
          cps_prob.append(0)
    # plot function
    cps_plot(cps_prob, curr_resp_label)

def main():
  file_name = 'nh_dbn.pckl'
  network = load_chain(file_name)
  network_configurations_list = network.network_configurations
  
  if file_name[:2] == 'fp': # check if the model is of the type full parents  
    print('Starting to plot for the full parents model...')
    t = time.time()

    # Betas bloxplots
    beta_boxplots_overtime(network.betas_over_time, network_configurations_list)

    # residual plots over-time
    residual_plots_overtime(network.betas_over_time, network.network_configurations)

    # Cps barplots
    chain_points_prob_plot(
      network.cps_over_response,
      network.network_configurations
    )

    ##### Line plot of fraction scores
    fraction_scores_plot(network.network_configurations, network.scores_over_time)

    print('Time elapsed: ', time.time() - t, ' seconds.')
  else: # the loaded model is not with full parents
    print('Starting to plot for the changing parents model...')
    t = time.time()
    # residual plots over-time
    residual_plots_overtime(network.betas_over_time, network.network_configurations)
    print('Time elapsed: ', time.time() - t, ' seconds.')

if __name__ == '__main__':
  main()
