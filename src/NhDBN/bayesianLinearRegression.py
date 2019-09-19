#from abc import ABC, abstractmethod
# Base class for the bayesian linear regression class

class BayesianLinearRegression():
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



