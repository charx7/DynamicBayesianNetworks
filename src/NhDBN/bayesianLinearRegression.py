#from abc import ABC, abstractmethod # not sure if abc is worth the overhead cost

class BayesianLinearRegression:
  '''
    Base class for all the bayesian linear regression estimator algorithms
    
    Attributes:
      data : dict of the current configuration of the data
        {
          'features':{
            'X1': numpy.ndarray,
          }
          'response':{
            'y': numpy.ndarray
          }
        }
      num_samples : int
        number of samples on the data
      num_iter : int
        number of iterations
      results : dict of str: list<float>
        dictionary containing the results of the chain 
  '''
  # Base class for the bayesian linear regression class
  def __init__(self, data, num_samples, num_iter = 5000):
    self.data = data
    self.num_samples = num_samples
    self.num_iter = num_iter
    self.results = None

  # @abstractmethod
  # def fit(self):
  #   '''
  #     Fit method to be implemented by the sub-class
  #   '''
  #   pass
  