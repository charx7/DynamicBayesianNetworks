import numpy as np
from marginalLikelihood import calculateMarginalLikelihood, calculateMarginalLikelihoodWithChangepoints
from priors import calculateFeatureSetPriorProb, calculateChangePointsSetPrior
from utils import constructNdArray, generateInitialFeatureSet, \
  constructMuMatrix, deleteMove, addMove, exchangeMove, selectData
from changepointMoves import cpBirthMove, cpRellocationMove, cpDeathMove
from random import randint


# This will propose and either accept or reject the move for the changepoints
def changepointsSetMove(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, pi, numSamples, it, change_points): 
  '''
    Function that proposes a move to a new changepoint set and either accepts
    it or rejects it based on a Metropolis Hashtings move

    Args:
      tdb: <2many>
        needs argument cleaning
    Returns:
      change_points : list<int>
        a list of integers containing the changepoints 

  '''
  # Calculate the marginal likelihood of the current cps set
  marginalTau = calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_gamma_sigma_sqr,
   beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, change_points)
  
  # Select a random birth, death or recllocate move
  randomInteger = randint(0,2)

  # Changepoint moves selection
  if randomInteger == 0: # If the random integer is 0 then do a birth move
    newChangePoints = cpBirthMove(change_points, numSamples)
    # Hashting ratio calculation
    hr = (numSamples - 1 - len(change_points)) / len(newChangePoints)

  elif randomInteger == 1: # do the death move
    try:
      newChangePoints = cpDeathMove(change_points)
    except ValueError: # If the func fail then we stay the same
      newChangePoints = change_points 
    # Hashtings ratio calculation
    hr = len(change_points) / (numSamples - 1 - len(newChangePoints))
    
  else: # do the rellocation move
    try:
      newChangePoints = cpRellocationMove(change_points)
    except ValueError: # If the func fail then we stay the same
      newChangePoints = change_points
    # Hashtings ratio calculation
    hr = 1

  # Calculate the marginal likelihood of the new cps set
  marginalTauStar = calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_gamma_sigma_sqr,
   beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, newChangePoints)
  
  # Prior calculations
  tauPrior = calculateChangePointsSetPrior(change_points)
  tauStarPrior = calculateChangePointsSetPrior(newChangePoints)

  # Get the threshhold of the probability of acceptance of the move
  acceptanceRatio = min(1, (marginalTauStar/marginalTau) * (tauStarPrior / tauPrior) * hr)
  # Get a sample from the U(0,1) to compare the acceptance ratio
  u = np.random.uniform(0,1)
  if u < acceptanceRatio:
    # if the sample is less than the acceptance ratio we accept the move to Tau* (the new cps)
    change_points = newChangePoints

  return change_points

# Switcher that defines what random move we are going to make
def selectMoveDict(selectedFunc):
  switcher = {
    0: addMove,
    1: deleteMove,
    2: exchangeMove
  }

  return switcher.get(selectedFunc)

def featureSetMoveWithChangePoints(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples, it, change_points):
  # The the possible features set
  possibleFeaturesSet = list(data['features'].keys())
  possibleFeaturesSet = [int(x.replace('X', '')) for x in possibleFeaturesSet]
  
  marginalPi = calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_gamma_sigma_sqr,
   beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, change_points)
  
  # Select a random add, delete or exchange move
  randomInteger = randint(0,2)
  func = selectMoveDict(randomInteger)
  # Try catch block for the random move
  try:
    piStar = func(pi, featureDimensionSpace, fanInRestriction, possibleFeaturesSet)
    # Construct the new X, mu
    partialData = {
      'features':{},
      'response':{}
    }
    for feature in piStar:
      currKey = 'X' + str(int(feature))
      partialData['features'][currKey] = data['features'][currKey]

    # Design Matrix Design tensor? Design NdArray?
    XStar = constructNdArray(partialData, numSamples, change_points) # TODO Fix the construction of piStar
    # Mu matrix
    muStar = constructMuMatrix(piStar) # TODO Verify the construction of muStar
    # Calculate marginal likelihook for PiStar
    marginalPiStar = calculateMarginalLikelihoodWithChangepoints(XStar, y, muStar,
     alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, change_points) 
    
  except ValueError:
    piStar = pi
    marginalPiStar = marginalPi

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
  
def featureSetMove(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples, it):
  # The the possible features set
  possibleFeaturesSet = list(data['features'].keys())
  possibleFeaturesSet = [int(x.replace('X', '')) for x in possibleFeaturesSet]
  # Calculate the probability of response given the feature set Pi (marginal likelihood)
  marginalPi = calculateMarginalLikelihood(X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples)    
  # Select a random add, delete or exchange move
  randomInteger = randint(0,2)
  
  func = selectMoveDict(randomInteger)
  # Try catch block for the random move
  try:
    piStar = func(pi, featureDimensionSpace, fanInRestriction, possibleFeaturesSet)
    # Construct the new X, mu
    partialData = {
      'features':{},
      'response':{}
    }
    for feature in piStar:
      currKey = 'X' + str(int(feature))
      partialData['features'][currKey] = data['features'][currKey]

    # Design Matrix
    XStar = constructDesignMatrix(partialData, numSamples) #TODO correct so this works again
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
  