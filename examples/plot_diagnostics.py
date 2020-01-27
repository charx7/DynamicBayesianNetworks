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

  sns.set(style="darkgrid") # style
  ax = sns.barplot(
    y = data,
    x = _x,
    palette="Blues_d",
    label="cps-prob-over-time").set(xlabel='time-points', ylabel='cps-prob')
  #plt.xlim(2, None) # show from 2 in case we want a line-plot
  plt.show()

def main():
  network = load_chain('fp_nh_dbn.pckl')

  #### TODO funcionalize this and do it for every response/edge
  # boxplots of just the first response y = 1
  curr_response_betas = network.betas_over_time[0]

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

  #### TODO functionalize this: Now the chainpoints prob vector
  # this is just the last chain so we need to store the other ones in the obj
  burned_cps = network.chain_results['tau_vector'][network.burn_in:]  
  thinned_changepoints = [burned_cps[x] for x in range(len(burned_cps)) if x%10==0]

  # get the vector of each cps probability
  cps_prob = []
  time_periods = len(network.network_configuration['response']['y'])
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

if __name__ == '__main__':
  main()
