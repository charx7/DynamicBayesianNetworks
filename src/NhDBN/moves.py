import numpy as np
from marginalLikelihood import calculateMarginalLikelihood
from priors import calculateFeatureSetPriorProb
from utils import constructDesignMatrix, generateInitialFeatureSet, constructMuMatrix, deleteMove, addMove, exchangeMove, selectData
from random import randint

# Switcher that defines what random move we are going to make
def selectMoveDict(selectedFunc):
  switcher = {
    0: addMove,
    1: deleteMove,
    2: exchangeMove
  }

  return switcher.get(selectedFunc)

def featureSetMove(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples, it):
  # Calculate the probability of response given the feature set Pi (marginal likelihood)
  marginalPi = calculateMarginalLikelihood(X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples)    
  # Select a random add, delete or exchange move
  randomInteger = randint(0,2)
  
  func = selectMoveDict(randomInteger)
  # Try catch block for the random move
  try:
    piStar = func(pi, featureDimensionSpace, fanInRestriction)
    # Construct the new X, mu
    partialData = {
      'features':{},
      'response':{}
    }
    for feature in piStar:
      currKey = 'X' + str(int(feature))
      partialData['features'][currKey] = data['features'][currKey]

    # Design Matrix
    XStar = constructDesignMatrix(partialData, numSamples)
    # Mu matrix
    muStar = constructMuMatrix(piStar)
    # Calculate marginal likelihook for PiStar
    marginalPiStar = calculateMarginalLikelihood(XStar, y, muStar, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples) 
    
  except ValueError:
    piStar = pi
    marginalPiStar = marginalPi
    # Calculate the probability of response the feature set Pi*
  
  # Calculate the prior probabilites of the move Pi -> Pi*
  piPrior = calculateFeatureSetPriorProb(pi, featureDimensionSpace, fanInRestriction) 
  piStarPrior = calculateFeatureSetPriorProb(piStar, featureDimensionSpace, fanInRestriction)

  # Calculate the acceptance/rejection probability of the move given Pi, Pi*
  # First we need to calculate HR given the move we selected
  if randomInteger == 0:
    # Add move
    hr = (featureDimensionSpace - len(pi)) / len(piStar)
  elif randomInteger == 1:
    # Delete Move
    hr = len(pi) / (featureDimensionSpace - len(piStar))
  elif randomInteger == 2:
    # Exchange move
    hr = 1
  # Get the threshhold of the probability of acceptance of the move
  acceptanceRatio = min(1, (marginalPiStar/marginalPi) * (piStarPrior/ piPrior) * hr)
  # Get a sample from the U(0,1) to compare the acceptance ratio
  u = np.random.uniform(0,1)
  if u < acceptanceRatio:
    # if the sample is less than the acceptance ratio we accept the move to Pi*
    pi = piStar

  return pi
  
if __name__ == '__main__':
  print('Test')
  