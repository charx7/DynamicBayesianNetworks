from bayesianPwLinearRegression import BayesianPieceWiseLinearRegression
from bayesianLinearRegression import BayesianLinearRegression
from seqCoupledBayesianPwLinReg import SeqCoupledBayesianPieceWiseLinearRegression
from scores import calculateFeatureScores, adjMatrixRoc

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
    network = self.data # retreive the network data
    dims = self.data.shape[1] # dimensions of the data points
    dimsVector = [x for x in range(dims)]
    num_samples = self.data.shape[0] # number of data points

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
      data_dict['features'][col_name] = network[:num_samples - 1, el]

      # Add the response to the dict
      data_dict['response']['y'] = network[1:, currResponse]

    self.network_configuration = data_dict # add the current config to the network

  def fit(self, method):
    '''
      Method that will the current data configuration of the network
      using the provided method (algorithm)

      Args:
        method : str
          string that will determine which method we are going to use 
    '''
    num_samples = self.data.shape[0] # Number of data points

    if method == 'varying_nh_dbn':   # call the nh-dbn with varying cps
      baReg = BayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config
        'varying_nh',                # varying changepoints non-homogeneous
        num_samples - 1,             # number of data points
        self.chain_length,           # len of chain
        [num_samples + 1]            # just the last pseudo cp []
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
        num_samples,                 # number of samples
        self.chain_length            # length of the MCMC chain
      )
      baReg.fit() # call to the fit method of the regressor
      self.chain_results = baReg.results # set the results
    elif method == 'seq_coup_nh_dbn':
      baReg = SeqCoupledBayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config
        'seq_coup_nh',               # varying changepoints non-homogeneous seq coupled
        num_samples - 1,             # number of data points
        self.chain_length,           # len of chain
        [num_samples + 1]            # just the last pseudo cp []
      )
      baReg.fit() # call the fit method of the regressor
      self.chain_results = baReg.results # set the results
      
  def score_edges(self, currResponse):
    '''
      Calculates de edge score for the current configuration of the network 

      Args:
        currResponse : int
          integer referencing which variable X_i is the 
          current response of the configuration
    '''
    dims = self.data.shape[1] # Get the number of features (dimensions of the data)

    currFeatures = [int(string[1]) for string in list(self.network_configuration['features'])]

    self.edge_scores = calculateFeatureScores(
        self.chain_results['pi_vector'][:self.burn_in],
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
    dims = self.data.shape[1] # dimensions of the data points
    dimsVector = [x for x in range(dims)]

    for configuration in dimsVector:
      self.set_network_configuration(configuration)
      self.fit(method)
      self.score_edges(configuration)
