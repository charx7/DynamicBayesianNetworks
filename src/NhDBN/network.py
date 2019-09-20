from pWLinRegNhdbn import pwGibbsSamplingWithMoves, pwGibbsSamplingWithCpsParentsMoves
from scores import calculateFeatureScores, adjMatrixRoc
from bayesianPwLinearRegression import BayesianPieceWiseLinearRegression

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
  def __init__(self, data, chain_length, burn_in):
    self.data = data
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

    if method == 'nh_dbn':
      baReg = BayesianPieceWiseLinearRegression(
        self.network_configuration,  # Current data config
        'nh',                        # non-homogeneous
        num_samples - 1,             # number of data points
        self.chain_length,           # len of chain
        [num_samples + 1]            # just the las pseudo cp []
      )
      baReg.fit() # Call the fit method of the regressor
      self.chain_results = baReg.results
    
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
