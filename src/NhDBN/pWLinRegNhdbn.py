import numpy as np
from scipy.stats import invgamma
from random import randint
from tqdm import tqdm
from utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, deleteMove, addMove, exchangeMove, selectData
from scores import calculateFeatureScores, drawRoc
from marginalLikelihood import calculateMarginalLikelihood
from priors import calculateFeatureSetPriorProb
from plotData import plotTrace, plotHistogram, plotScatter
from generateTestData import generateTestDataSecond

def testAlgorithm():
  # Set Seed
  np.random.seed(42)
  # Generate data to test our algo
  num_samples = 100
  dims = 6
  data = generateTestDataSecond(num_samples = num_samples, dimensions = dims)

  print(data)
  # Do the gibbs Sampling
  
if __name__ == '__main__':
  testAlgorithm()
