from .bayesianPwLinearRegression import BayesianPieceWiseLinearRegression
from .bayesianLinearRegression import BayesianLinearRegression
from .seqCoupledBayesianPwLinReg import SeqCoupledBayesianPieceWiseLinearRegression
from .globCoupBayesianPwLinReg import GlobCoupledBayesianPieceWiseLinearRegression
from .scores import calculateFeatureScores, adjMatrixRoc, credible_interval, \
 credible_score, get_betas_over_time, get_scores_over_time, beta_post_matrix, score_beta_matrix
from .fullParentsBpwLinReg import FPBayesianPieceWiseLinearRegression
from .fpBayesianLinearRegression import FpBayesianLinearRegression
from .fpGlobCoupBpwLinReg import FpGlobCoupledBayesianPieceWiseLinearRegression
from .fpSeqCoupBpwlinReg import FpSeqCoupledBayesianPieceWiseLinearRegression
import numpy as np

class Network():
  '''
    Class that serves as the waypoint to infer the network topology
    of a dataset using various different implemented algorithms

    Attributes:
      data : numpy.ndarray
        numpy array with shape (num_samples, variables)
      chain_length : int
        integer containing the chain length
      burn_in : int
        integer that determines the burn_in interval of the MCMC chain 
  '''
  def __init__(self, data, chain_length, burn_in, change_points = []):
    self.data = data
    self.change_points = change_points
    self.network_configuration = None
    self.chain_length = chain_length 
    self.burn_in = burn_in
    #self.method = 'nh_dbn'
    self.true_adj_matrix = None
    self.proposed_adj_matrix = [] # proposed adj matrix
    self.edge_scores = None
    self.chain_results = None
    self.scores_over_time = [] # scores over time list of matrices
    self.betas_over_time = [] # we also want the betas over time for diagnostics
    self.cps_over_response = [] # we want all the different computed chains
    self.network_configurations = [] # the list of all the design matrices of all the network configs

  def set_network_configuration(self, configuration):
    '''
      Method transforms and sets the 'raw' data using the 
      given configuration into a dictionary of the form:
      {
        'features': {
          'X1': numpy.ndarray
          ...
        }
        'response': {
          'y': numpy.ndarray
        }
      }
      
      Args:
        configuration : int
          integer that indicates which variable X_i is the current response
    '''    
    network_list = self.data # retreive the network data
    dims = self.data[0].shape[1] # dimensions of the data points
    dimsVector = [x for x in range(dims)]
    
    num_samples = 0 
    for segment in network_list:
      # add the length of the segment
      num_samples = segment.data.shape[0] + num_samples
      
    #num_samples = self.data.shape[0] # number of data points

    currResponse = configuration # Which column will be the response for the configuration
    # You have to evaluate because the filter returns an obj
    currFeatures = list(filter(lambda x: x != configuration, dimsVector))

    data_dict = {
      'features': {},
      'response': {}
    }

    # Add the features to the dict
    for el in currFeatures:
      col_name = 'X' + str(el)
      feature_data = np.array([]) # data initilize as empty
      for segment in network_list:
        curr_segment_len = segment.shape[0]
        # select all but the last data point
        segment_data = segment[:curr_segment_len - 1, el]
        # concatenate(stack) the segment data into the data of the curr feature
        feature_data = np.concatenate((feature_data, segment_data)) if feature_data.size else segment_data

      # add to the dict
      data_dict['features'][col_name] = feature_data

    # Select + stack the data for the response
    resp_data = np.array([]) # resp init as empty
    for segment in network_list:
      curr_resp_len = segment.shape[0]
      segment_data = segment[1:curr_resp_len, currResponse] # select curr resp data
      # concatenate the resp data
      resp_data = np.concatenate((resp_data, segment_data), axis = 0) if resp_data.size else segment_data

    data_dict['response']['y'] = resp_data
    
    self.network_configuration = data_dict # add the current config to the network
    self.network_configurations.append(data_dict) # append the current network config 

  def fit(self, method):
    '''
      Method that will the current data configuration of the network
      using the provided method (algorithm)

      Args:
        method : str
          string that will determine which method we are going to use 
    '''
    num_samples = self.network_configuration['response']['y'].shape[0] # Number of data points

    if method == 'varying_nh_dbn':   # call the nh-dbn with varying cps
      baReg = BayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config
        'varying_nh',                # varying changepoints non-homogeneous
        num_samples,                 # number of data points
        self.chain_length,           # len of chain
        [num_samples + 2]            # just the last pseudo cp []
      )
      baReg.fit() # Call the fit method of the regressor
      self.chain_results = baReg.results # Set the results
    elif method == 'fp_varying_nh_dbn': # full parents credible intervals method
      baReg = FPBayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config
        'varying_nh',                # varying changepoints non-homogeneous
        num_samples,                 # number of data points
        self.chain_length,           # len of chain
        [num_samples + 2]            # just the last pseudo cp []
      )
      baReg.fit() # Call the fit method of the regressor
      self.chain_results = baReg.results # Set the results
    elif method == 'fixed_nh_dbn':   # call the nh-dbn with fixed cps
      baReg = BayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config of the network
        'fixed_nh',                  # fixed cps non-homogeneous
        num_samples - 1,             # number of data points
        self.chain_length,           # length of the MCMC
        self.change_points           # predefined cps 
      )
      baReg.fit() # call the fit method of the regressor
      self.chain_results = baReg.results # set the results
    elif method == 'h_dbn':          # call the h-dbn
      baReg = BayesianLinearRegression(
        self.network_configuration,  # current data config of the network
        num_samples + 1,             # number of samples
        self.chain_length            # length of the MCMC chain
      )
      baReg.fit() # call to the fit method of the regressor
      self.chain_results = baReg.results # set the results
    elif method == 'fp_h_dbn':       # call the full parents h-dbn
      baReg = FpBayesianLinearRegression(
        self.network_configuration,  # current data config of the network
        num_samples + 1,             # number of samples
        self.chain_length            # length of the MCMC chain
      )
      baReg.fit() # call the fit method of the regressor
      self.chain_results = baReg.results # set the results
    elif method == 'seq_coup_nh_dbn':
      baReg = SeqCoupledBayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config
        'seq_coup_nh',               # varying changepoints non-homogeneous seq coupled
        num_samples - 1,             # number of data points
        self.chain_length,           # len of chain
        [num_samples + 2]            # just the last pseudo cp []
      )
      baReg.fit() # call the fit method of the regressor
      self.chain_results = baReg.results # set the results
    elif method == 'fp_seq_coup_nh_dbn':
      baReg = FpSeqCoupledBayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config
        'seq_coup_nh',               # varying changepoints non-homogeneous seq coupled
        num_samples - 1,             # number of data points
        self.chain_length,           # len of chain
        [num_samples + 2]            # just the last pseudo cp []
      )
      baReg.fit() # call the fit method of the regressor
      self.chain_results = baReg.results # set the results
    elif method == 'glob_coup_nh_dbn':
      baReg = GlobCoupledBayesianPieceWiseLinearRegression(
        self.network_configuration,   # current data config
        'glob_coup_nh',               # glob coup additional functions
        num_samples,                  # number of data points
        self.chain_length,            # len of chain
        [num_samples + 2]             # just the last pseudo cp []
      )
      baReg.fit() # call to the fit method of the glob coup regressor
      self.chain_results = baReg.results
    elif method == 'fp_glob_coup_nh_dbn':
      baReg = FpGlobCoupledBayesianPieceWiseLinearRegression(
        self.network_configuration,   # current data config
        'glob_coup_nh',               # glob coup additional functions
        num_samples,                  # number of data points
        self.chain_length,            # length of the chain
        [num_samples + 2]             # just the last pseudo cp []
      )
      baReg.fit() # call to the fit method of the glob coup regressor
      self.chain_results = baReg.results

  def score_edges(self, currResponse, method):
    '''
      Calculates de edge score for the current configuration of the network 

      Args:
        currResponse : int
          integer referencing which variable X_i is the 
          current response of the configuration
        method : str
          string that contains the type of method used so we can evaluate 
          with the chain_results of the pi_vector or with the credible intervals
          for the full parent sets
    '''
    dims = self.data[0].shape[1] # dimensions of the data points
    currFeatures = [int(string[1]) for string in list(self.network_configuration['features'])]

    # check if the method is for full parents
    # this should only check the first 2 letters of the method
    if (method == 'fp_varying_nh_dbn' 
      or method == 'fp_h_dbn'
      or method == 'fp_seq_coup_nh_dbn'
      or method == 'fp_glob_coup_nh_dbn'): 
      
      # thin + burn the chain on the global mean chain
      if method == 'fp_glob_coup_nh_dbn':
        # if the method is from the glob coup we will use the global mean vector 
        burned_chain = self.chain_results['mu_vector'][self.burn_in:]
        thinned_chain = [burned_chain[x] for x in range(len(burned_chain)) if x%10==0]
        # necessary so the beta matrix is built correctly
        mu_thinned_chain = [[element] for element in thinned_chain]
      
      # shift the betas by 2 so it fits with the cps
      betas_chain = self.chain_results['betas_vector'][2:]
      # burn the shifted chain
      burned_chain = betas_chain[self.burn_in:]
      # thin both of the chains
      betas_thinned_chain = [burned_chain[x] for x in range(len(burned_chain)) if x%10==0]
      
      # we only have chainpoints in the non-homogeneous model
      if method != 'fp_h_dbn':
        # burn the cps chain
        burned_cps = self.chain_results['tau_vector'][self.burn_in:] 
        thinned_changepoints = [burned_cps[x] for x in range(len(burned_cps)) if x%10==0]
      else:
        thinned_changepoints = [] # if not then we just assign an empty list

      # This will get the betas over time as diagnostic
      if (method == 'fp_varying_nh_dbn'
        or method == 'fp_seq_coup_nh_dbn'
        or method == 'fp_glob_coup_nh_dbn'): 
        # get the len of the time-series
        time_pts = self.network_configuration['response']['y'].shape[0]
        betas_over_time = get_betas_over_time(time_pts, thinned_changepoints, betas_thinned_chain, dims) #TODO add the dims
        self.betas_over_time.append(betas_over_time) # append to the network
        scores_over_time = get_scores_over_time(betas_over_time, currFeatures, dims)
        self.scores_over_time.append(scores_over_time) # append to the network

      if method == 'fp_glob_coup_nh_dbn':
        # if we are using a glob coup model change the scores to the global vector
        # TODO make this a user input
        betas_thinned_chain = mu_thinned_chain

      betas_matrix = beta_post_matrix(betas_thinned_chain) # construct the betas post matrix
      edge_scores = score_beta_matrix(betas_matrix, currFeatures, currResponse) # score the matrix
      self.proposed_adj_matrix.append(edge_scores) # append to the proposed adj matrix
      self.cps_over_response.append(thinned_changepoints) # append the cps chain over the curr response
    else:
      # burn + thin the chain
      burned_chain = self.chain_results['pi_vector'][self.burn_in:]
      thinned_chain =  [burned_chain[x] for x in range(len(burned_chain)) if x%10!=0]

      self.edge_scores = calculateFeatureScores(
          #self.chain_results['pi_vector'][self.burn_in:],
          thinned_chain,
          dims, 
          currFeatures,
          currResponse)

      self.proposed_adj_matrix.append(self.edge_scores) # append to the proposed adj matrix

  def infer_network(self, method):
    '''
      Infers the network topology on the data by changing to all
      possible configurations of the network 

      Args:
        method : str
          string with the name of the method we are going to use 
          to fit the data
    '''
    # because data is now a list we have to select the first allways
    # existing element
    dims = self.data[0].shape[1] # dimensions of the data points
    dimsVector = [x for x in range(dims)]

    for configuration in dimsVector:
      self.set_network_configuration(configuration)
      self.fit(method)
      self.score_edges(configuration, method)
