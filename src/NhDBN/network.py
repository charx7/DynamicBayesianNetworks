from pWLinRegNhdbn import pwGibbsSamplingWithMoves, pwGibbsSamplingWithCpsParentsMoves
from scores import calculateFeatureScores, adjMatrixRoc
from bayesian_pw_Lin_reg import BayesianPwLinearRegression

class Network():
  def __init__(self, data, chain_length, burn_in):
    self.data = data
    self.network_configuration = None
    self.chain_length = chain_length 
    self.burn_in = burn_in
    self.method = 'nh_dbn'
    self.true_adj_matrix = None
    self.proposed_adj_matrix = [] # proposed adj matrix
    self.edge_scores = None
    self.chain_results = None
    
  def set_network_configuration(self, configuration):    
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

  def fit(self):
    num_samples = self.data.shape[0] # Number of data points

    if self.method == 'nh_dbn':
      baReg = BayesianPwLinearRegression(
        self.network_configuration, 
        [num_samples + 1],
        num_samples - 1,
        self.chain_length
      )
      baReg.fit() # Call the fit method of the regressor
      self.chain_results = baReg.results
    
  def score_edges(self, currResponse):
    dims = self.data.shape[1] # Get the number of features (dimensions of the data)

    currFeatures = [int(string[1]) for string in list(self.network_configuration['features'])]

    self.edge_scores = calculateFeatureScores(
        self.chain_results['pi_vector'][:self.burn_in],
        dims, 
        currFeatures,
        currResponse)

    self.proposed_adj_matrix.append(self.edge_scores) # append to the proposed adj matrix

  def infer_network(self):
    dims = self.data.shape[1] # dimensions of the data points
    dimsVector = [x for x in range(dims)]

    for configuration in dimsVector:
      self.set_network_configuration(configuration)
      self.fit()
      self.score_edges(configuration)
