import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from utils import load_chain

def boxplot(data):
  #### With matplotlib
  # # Create a figure instance
  # fig = plt.figure(1, figsize=(9, 6))

  # # Create an axes instance
  # ax = fig.add_subplot(111)

  # # Create the boxplot
  # box_plot = ax.boxplot(data)

  # plt.show() # show
  
  #### With seaborn
  title = 'Edge 2 -> 1 Boxplot Betas Values Across Time'
  col_names = [i for i in range(data.shape[1] + 1)]
  col_names = [i for i in col_names if i > 0] 
  sns.set(style="darkgrid") # style
  df = pd.DataFrame(data, columns= col_names)
  ax = sns.boxplot(data=df).set(xlabel='time-points', ylabel='beta-value', title = title)
 
  # Add jitter with the swarmplot function.
  #ax = sns.swarmplot(data=df, color="grey") # seems too clunky
  plt.show()

def cps_plot(data):
  _x = [i for i in range(33)]
  _x = [i for i in _x if i > 1]

  title = 'Cps Probability Across Time of the First Chain Configuration'
  sns.set(style="darkgrid") # style
  ax = sns.barplot(
    y = data,
    x = _x,
    palette="Blues_d",
    label="cps-prob-over-time").set(xlabel='time-points', ylabel='cps-prob')
  #plt.xlim(2, None) # show from 2 in case we want a line-plot
  plt.title(title)
  plt.show()

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
  plt.show()

def fraction_scores_plot(data):
  # TODO this needs to be called for every configuration of the data
  curr_network_configuration = data[0] # the first config for now

  # this needs to be done over curr_network_configuration.shape[1] this case 4 edges
  curr_edge = curr_network_configuration[:,0]
  line_plot(curr_edge, '2', '1')

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

# TODO join this and the other boxplot into one func
def boxplot_res(data):
  title = 'Residuals over time of response Y = X0'
  col_names = [i for i in range(data.shape[1] + 1)]
  col_names = [i for i in col_names if i > 0] 
  sns.set(style="darkgrid") # style
  df = pd.DataFrame(data, columns= col_names)
  ax = sns.boxplot(data=df).set(xlabel='time-points', ylabel='residual-value', title = title)
 
  # Add jitter with the swarmplot function.
  #ax = sns.swarmplot(data=df, color="grey") # seems too clunky
  plt.show()

def residual_plots_overtime(betas_response_list, network_configs):
  m_list = get_design_matrices_list(network_configs) # get the list of design matrices
  r_list = get_response_list(network_configs) # get the list of response matrices

  # get the first el from the list of configs
  curr_matrix_config = m_list[0]

  # get the betas for the fist config
  curr_response_betas = betas_response_list[0]
  # get the responses for the first config
  curr_response_list = r_list[0]

  curr_residuals_matrix = np.array([])
  # get the betas for every time point
  for time, curr_time_pt_betas in enumerate(curr_response_betas):
    curr_x = curr_matrix_config[time] # get the x of the curr time point
    curr_x = curr_x.reshape(-1,1) # reshape + transpose 

    preds = np.dot(curr_time_pt_betas, curr_x) # compute pred values
    resp = curr_response_list[time] # get the current resp

    residuals = resp - preds # compute the residuals vector
    
    # horizontally stack them into a matrix
    curr_residuals_matrix = np.hstack((curr_residuals_matrix, residuals)) if curr_residuals_matrix.size else residuals

  # plot it!
  boxplot_res(curr_residuals_matrix)

def beta_boxplots_overtime(betas_response_list):
  #### TODO Do it for every response/edge
  # boxplots of just the first response y = 1
  curr_response_betas = betas_response_list[0]
  
  curr_response_matrix = np.array([])
  # get the betas for every time point
  for curr_time_pt_betas in curr_response_betas:
    # betas for the current edge
    curr_edge_betas = curr_time_pt_betas[:,0] # TODO here the 0 should be and idx for a different edge
    curr_edge_betas = curr_edge_betas.reshape(curr_edge_betas.shape[0] , 1) # reshape for the horizontal stack
    # horizontally stack them into a matrix
    curr_response_matrix = np.hstack((curr_response_matrix, curr_edge_betas)) if curr_response_matrix.size else curr_edge_betas

  # plot it!
  boxplot(curr_response_matrix)

def chain_points_prob_plot(cps_over_response, network_config):
  #### Now the chainpoints prob vector
  # TODO get the chain values for the first response configuration this has to be inside a loop
  thinned_changepoints = cps_over_response[0] # 0 has to be an idx

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
  cps_plot(cps_prob)

def main():
  network = load_chain('fp_nh_dbn.pckl')

  # Betas bloxplots
  beta_boxplots_overtime(network.betas_over_time)

  # residual plots over-time
  residual_plots_overtime(network.betas_over_time, network.network_configurations)

  # Cps barplots
  chain_points_prob_plot(
    network.cps_over_response,
    network.network_configuration['response']['y']
  )

  ##### Line plot of fraction scores
  fraction_scores_plot(network.scores_over_time)

if __name__ == '__main__':
  main()
