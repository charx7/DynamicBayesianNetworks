import numpy as np
import math
from scipy.stats import multivariate_normal
from random import randint

from .marginalLikelihood import calculateMarginalLikelihood, calculateMarginalLikelihoodWithChangepoints
from .priors import calculateFeatureSetPriorProb, calculateChangePointsSetPrior
from .utils import constructNdArray, generateInitialFeatureSet, \
  constructMuMatrix, deleteMove, addMove, exchangeMove, selectData, \
  constructNdArray, constructMuMatrix, constructResponseNdArray, \
  constructDesignMatrix
from .changepointMoves import cpBirthMove, cpRellocationMove, cpDeathMove
from .samplers import muSampler, betaTildeSampler

def globCoupChangepointsSetMove(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, sigma_sqr, pi, numSamples, it, change_points, method = '', delta_sqr = []):
  '''
    Documentation TODO
  '''
  try: # get the value of delta sqr
    curr_delta_sqr = delta_sqr[it + 1]
  except IndexError: # we are in a method that does not require delta^2
    curr_delta_sqr = []

  # Select a random birth, death or rellocate move
  randomInteger = randint(0,2)

  # Changepoint moves selection
  validMove = True
  if randomInteger == 0: # If the random integer is 0 then do a birth move
    newChangePoints = cpBirthMove(change_points, numSamples)
    if len(newChangePoints) > 9:
      validMove = False
    else:
      # Hashting ratio calculation
      hr = (numSamples - 1 - len(change_points)) / (len(newChangePoints))

  elif randomInteger == 1: # do the death move
    try:
      newChangePoints = cpDeathMove(change_points)
    except ValueError: # If the func fail then we stay the same
      validMove = False
      newChangePoints = change_points 
    # Hashtings ratio calculation
    hr = (len(change_points)) / (numSamples - 1 - len(newChangePoints))
    
  else: # do the rellocation move
    try:
      newChangePoints = cpRellocationMove(change_points)
    except ValueError: # If the func fail then we stay the same
      validMove = False
      #newChangePoints = change_points
    # Hashtings ratio calculation
    hr = 1
  if validMove:
    # Calculate the marginal likelihood of the current cps set
    logmarginalTau = calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_gamma_sigma_sqr,
    beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, 
    change_points, method, curr_delta_sqr)
    
    # Get the density of mu
    _, muDensity = muSampler(mu, change_points,
      X, y, sigma_sqr[it + 1], lambda_sqr[it + 1])

    # ---> Reconstruct the design ndArray, mu vector and parameters for the marg likelihook calc
    # Select the data according to the set Pi
    partialData = selectData(data, pi)
    # Design ndArray
    XStar = constructNdArray(partialData, numSamples, newChangePoints)
    respVector = data['response']['y'] # We have to partition y for each changepoint as well
    yStar = constructResponseNdArray(respVector, newChangePoints)
    # Mu matrix 
    muDagger = constructMuMatrix(pi) 
  
    # Mu matrix star matrix (new)
    muStar, muStarDensity = muSampler(muDagger, newChangePoints, 
      XStar, yStar, sigma_sqr[it + 1], lambda_sqr[it + 1])

    # After changes on the design matrix now we can calculate the modified marg likelihood
    # Calculate the marginal likelihood of the new cps set and new mu star
    logmarginalTauStar = calculateMarginalLikelihoodWithChangepoints(XStar, yStar, muStar, 
    alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, 
    newChangePoints, method, curr_delta_sqr)

    # Prior calculations for tau, tau*, mu, mu*
    # >>>>>>>>>>>>>>>>>>
    tauPrior = calculateChangePointsSetPrior(change_points)
    tauStarPrior = calculateChangePointsSetPrior(newChangePoints)

    # Calculate the prior probabilities for mu and mu*
    # TODO functionalize this calculations
    muDagger = np.zeros(mu.shape[0])
    muDaggerPlus = np.zeros(muStar.shape[0]) # we need this in order to calc the density
    sigmaDagger = np.eye(muDagger.shape[0])
    sigmaDaggerPlus = np.eye(muDaggerPlus.shape[0])
    muStarPrior = multivariate_normal.pdf(muStar.flatten(), mean = muDaggerPlus.flatten(),
      cov = sigmaDaggerPlus)
    muPrior = multivariate_normal.pdf(mu.flatten(), mean = muDagger.flatten(),
      cov = sigmaDagger)

    # Get the threshhold of the probability of acceptance of the move
    acceptanceRatio = min(1,
      logmarginalTauStar - logmarginalTau + 
      math.log(tauStarPrior) - math.log(tauPrior) +
      math.log(muStarPrior) - math.log(muPrior) + 
      math.log(muDensity) - math.log(muStarDensity) + math.log(hr)
    )
    # Get a sample from the U(0,1) to compare the acceptance ratio
    u = np.random.uniform(0,1)
    if u < math.exp(acceptanceRatio):
      # if the sample is less than the acceptance ratio we accept the move to Tau* (the new cps)
      change_points = newChangePoints
      # also move to mu*
      mu = muStar

  return change_points, mu

# This will propose and either accept or reject the move for the changepoints
def changepointsSetMove(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, pi, numSamples, it, change_points, method = '', delta_sqr = []): 
  '''
    Function that proposes a move to a new changepoint set and either accepts
    it or rejects it based on a Metropolis Hashtings move

    Args:
      tdb : <2many>
        needs argument cleaning
        
    Returns:
      change_points : list<int>
        a list of integers containing the changepoints 

  '''
  try: # get the value of delta sqr
    curr_delta_sqr = delta_sqr[it + 1]
  except IndexError: # we are in a method that does not require delta^2
    curr_delta_sqr = []

  # Select a random birth, death or recllocate move
  randomInteger = randint(0,2)

  # Changepoint moves selection
  validMove = True
  if randomInteger == 0: # If the random integer is 0 then do a birth move
    newChangePoints = cpBirthMove(change_points, numSamples)
    if len(newChangePoints) == 10: # cannot go beyond 10 cps
      validMove = False
    # Hashting ratio calculation
    hr = (numSamples - len(change_points)) / (len(newChangePoints) -1)
  elif randomInteger == 1: # do the death move
    try:
      newChangePoints = cpDeathMove(change_points)
      hr = (len(change_points) - 1) / (numSamples - len(newChangePoints))
    except ValueError: # If the func fail then we stay the same
      newChangePoints = change_points
      validMove = False # not a valid move
      hr = 0
  else: # do the rellocation move
    try:
      newChangePoints = cpRellocationMove(change_points)
      hr = 1
    except ValueError: # If the func fail then we stay the same
      newChangePoints = change_points
      validMove = False
      hr = 0

  # Do the computations only if the changepoint move was a valid move
  if validMove:
    if method == 'seq-coup':
      # if the method is seq coupled then we calculate beta tildes vector
      muSeq = betaTildeSampler(y, X, mu, change_points,
       lambda_sqr[it + 1], delta_sqr[it + 1])
      # Calculate the marginal likelihood of the current cps set
      logmarginalTau = calculateMarginalLikelihoodWithChangepoints(X, y, muSeq, alpha_gamma_sigma_sqr,
      beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, 
      change_points, method, curr_delta_sqr)
    else:
      # Calculate the marginal likelihood of the current cps set
      logmarginalTau = calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_gamma_sigma_sqr,
      beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, 
      change_points, method, curr_delta_sqr)
    
    # ---> Reconstruct the design ndArray, mu vector and parameters for the marg likelihook calc
    # Select the data according to the set Pi
    partialData = selectData(data, pi)
    # Design ndArray
    XStar = constructNdArray(partialData, numSamples, newChangePoints)
    respVector = data['response']['y'] # We have to partition y for each changepoint as well
    yStar = constructResponseNdArray(respVector, newChangePoints)
    # Mu matrix
    muStar = constructMuMatrix(pi) 

    if method == 'seq-coup':
      # if the method is seq coupled then we calculate beta tildes vector
      muSeqStar = betaTildeSampler(yStar, XStar, muStar, newChangePoints,
       lambda_sqr[it + 1], delta_sqr[it + 1])

      # Calculate the marginal likelihood of the new cps set
      logmarginalTauStar = calculateMarginalLikelihoodWithChangepoints(XStar, yStar, muSeqStar, 
      alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, 
      newChangePoints, method, curr_delta_sqr)
    else:
      # Calculate the marginal likelihood of the new cps set
      logmarginalTauStar = calculateMarginalLikelihoodWithChangepoints(XStar, yStar, muStar, 
      alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, 
      newChangePoints, method, curr_delta_sqr)

    # Prior calculations
    tauPrior = calculateChangePointsSetPrior(change_points)
    logtauPrior = math.log(tauPrior)
    tauStarPrior = calculateChangePointsSetPrior(newChangePoints)
    logtauStarPrior = math.log(tauStarPrior)

    # Get the threshhold of the probability of acceptance of the move
    acceptanceRatio = min(1, math.exp(logmarginalTauStar - logmarginalTau + logtauStarPrior - logtauPrior + math.log(hr)))
    # acceptanceRatio = min(1, math.exp(
    #   math.log(marginalTauStar) - math.log(marginalTau) +
    #   math.log(tauStarPrior) - math.log(tauPrior) + math.log(hr)
    # ))
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

def globCoupFeatureSetMoveWithChangePoints(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, sigma_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples,
  it, change_points, method = '', delta_sqr = []):
  '''
    Documentation is missing for this function
  '''
  # TODO fix this redundant code delta param is not used here
  try: # get the value of the current delta
    curr_delta_sqr =  delta_sqr[it + 1]
  except IndexError: # we are not in a method that requires delta^2
    curr_delta_sqr = [] 
  
  # Get the possible features set
  possibleFeaturesSet = list(data['features'].keys())
  possibleFeaturesSet = [int(x.replace('X', '')) for x in possibleFeaturesSet]
  
  # Select a random add, delete or exchange move
  randomInteger = randint(0,2)
  # Do the calculation according to the randomly selected move
  validMove = True
  if randomInteger == 0:
    # Add move
    try:
      piStar = addMove(pi, featureDimensionSpace, fanInRestriction, possibleFeaturesSet)
      hr = (featureDimensionSpace - len(pi)) / len(piStar) # HR calculation
    except ValueError:
      validMove = False
      piStar = pi
      hr = 0
  elif randomInteger == 1:
    # Delete Move
    try:
      piStar = deleteMove(pi, featureDimensionSpace, fanInRestriction, possibleFeaturesSet)
      hr = len(pi) / (featureDimensionSpace - len(piStar)) # HR calculation
    except ValueError:
      validMove = False
      piStar = pi
      hr = 0
  elif randomInteger == 2:
    # Exchange move
    try:
      piStar = exchangeMove(pi, featureDimensionSpace, fanInRestriction, possibleFeaturesSet)
      hr = 1
    except ValueError:
      validMove = False
      piStar = pi
      hr = 0
  if validMove:
    # Construct the new X, mu
    partialData = {
      'features':{},
      'response':{}
    }
    for feature in piStar:
      currKey = 'X' + str(int(feature))
      partialData['features'][currKey] = data['features'][currKey]

    # Design Matrix Design tensor? Design NdArray?
    XStar = constructNdArray(partialData, numSamples, change_points)
    muDagger = constructMuMatrix(piStar) 
    # Mu matrix star matrix (new)
    muStar, muStarDensity = muSampler(muDagger, change_points, 
      XStar, y, sigma_sqr[it + 1], lambda_sqr[it + 1])

    # Calculate marginal likelihook for PiStar
    logmarginalPiStar = calculateMarginalLikelihoodWithChangepoints(XStar, y, muStar,
      alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples,
      change_points, method, curr_delta_sqr) 
    # Calculate marginal likelihood for Pi
    logmarginalPi = calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_gamma_sigma_sqr,
    beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, change_points, method, curr_delta_sqr)

    # Calculate mu density
    _, muDensity = muSampler(mu, change_points, 
      X, y, sigma_sqr[it + 1], lambda_sqr[it + 1])

    # Calculate the prior probabilites of the move Pi -> Pi*
    piPrior = calculateFeatureSetPriorProb(pi, featureDimensionSpace, fanInRestriction) 
    logpiPrior = math.log(piPrior)
    piStarPrior = calculateFeatureSetPriorProb(piStar, featureDimensionSpace, fanInRestriction)
    logpiStarPrior = math.log(piStarPrior)

    # Calculate the prior probabilities for mu and mu*
    # TODO functionalize this calculations
    muDagger = np.zeros(mu.shape[0])
    muDaggerPlus = np.zeros(muStar.shape[0]) # we need this in order to calc the density
    sigmaDagger = np.eye(muDagger.shape[0])
    sigmaDaggerPlus = np.eye(muDaggerPlus.shape[0])
    muStarPrior = multivariate_normal.pdf(muStar.flatten(), mean = muDaggerPlus.flatten(),
      cov = sigmaDaggerPlus)
    muPrior = multivariate_normal.pdf(mu.flatten(), mean = muDagger.flatten(),
      cov = sigmaDagger)

    # Calculate the final acceptance probability A(~) TODO this should be in log 
    # terms to avoid underflow with very low densities!!!
    acceptanceRatio = min(1,
    logmarginalPiStar - logmarginalPi + 
    logpiStarPrior - logpiPrior + math.log(muDensity) - math.log(muStarDensity) + 
    math.log(muStarPrior) - math.log(muPrior) + math.log(hr)
    )

    # Get a sample from the U(0,1) to compare the acceptance ratio
    u = np.random.uniform(0,1)
    if u < math.exp(acceptanceRatio):
      # if the sample is less than the acceptance ratio we accept the move to Pi*
      X = XStar
      pi = piStar
      mu = muStar

  return pi, mu, X

def featureSetMoveWithChangePoints(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples,
  it, change_points, method = '', delta_sqr = []):
  '''
    Documentation is missing for this function
  '''
  # The possible features set
  possibleFeaturesSet = list(data['features'].keys())
  possibleFeaturesSet = [int(x.replace('X', '')) for x in possibleFeaturesSet]
  
  try: # get the value of the current delta
    curr_delta_sqr =  delta_sqr[it + 1]
  except IndexError: # we are not in a method that requires delta^2
    curr_delta_sqr = [] 
      
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

    if method == 'seq-coup':
      # if the method is seq coupled then we calculate beta tildes vector
      muStarSeq = betaTildeSampler(y, XStar, muStar, change_points,
       lambda_sqr[it + 1], delta_sqr[it + 1])
      # Calculate seq coup marginal likelihook for PiStar
      logmarginalPiStar = calculateMarginalLikelihoodWithChangepoints(XStar, y, muStarSeq,
      alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples,
      change_points, method, curr_delta_sqr) 
    else:
      # Calculate marginal likelihook for PiStar
      logmarginalPiStar = calculateMarginalLikelihoodWithChangepoints(XStar, y, muStar,
      alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples,
      change_points, method, curr_delta_sqr) 
    
    validMove = True # the selected move was valid
  
  except ValueError:
    piStar = pi
    #marginalPiStar = 0
    marginalPiStar = 1 # we are using the logs
    validMove = False # we had a invalid move

  # for efficiency we will only do the computations when we selected a valid move
  if validMove:
    if method == 'seq-coup':
      # if the method is seq coupled then we calculate beta tildes vector
      muSeq = betaTildeSampler(y, X, mu, change_points,
       lambda_sqr[it + 1], delta_sqr[it + 1])

      # seq coup marginal likelihood computation with pi
      logmarginalPi = calculateMarginalLikelihoodWithChangepoints(X, y, muSeq, alpha_gamma_sigma_sqr,
      beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, change_points, method, curr_delta_sqr)
    
    else:
      # marginal likelihood computation with pi
      logmarginalPi = calculateMarginalLikelihoodWithChangepoints(X, y, mu, alpha_gamma_sigma_sqr,
      beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples, change_points, method, curr_delta_sqr)

    # Calculate the prior probabilites of the move Pi -> Pi*
    piPrior = calculateFeatureSetPriorProb(pi, featureDimensionSpace, fanInRestriction) 
    logpiPrior = math.log(piPrior)
    piStarPrior = calculateFeatureSetPriorProb(piStar, featureDimensionSpace, fanInRestriction)
    logpiStarPrior = math.log(piStarPrior)

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
    acceptanceRatio = min(1, math.exp(logmarginalPiStar - logmarginalPi  + logpiStarPrior - logpiPrior + math.log(hr)))
    # Now we do the log to avoid underflow
    # acceptanceRatio = min( \
    #   1, \
    #   math.exp(
    #     math.log(marginalPiStar) - 
    #     math.log(marginalPi) +
    #     math.log(piStarPrior) -
    #     math.log(piPrior) +
    #     math.log(hr)
    # ))
    # Get a sample from the U(0,1) to compare the acceptance ratio
    u = np.random.uniform(0,1)
    if u < acceptanceRatio:
      # if the sample is less than the acceptance ratio we accept the move to Pi*
      pi = piStar
      X = XStar
      mu = muStar

  return pi, X, mu
  
def featureSetMove(data, X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr,
  lambda_sqr, pi, fanInRestriction, featureDimensionSpace, numSamples, it):
  '''
    Documentation is missing for this function
  '''
  # The the possible features set
  possibleFeaturesSet = list(data['features'].keys())
  possibleFeaturesSet = [int(x.replace('X', '')) for x in possibleFeaturesSet]
  
  # Select a random add, delete or exchange move
  randomInteger = randint(0,2)
  
  func = selectMoveDict(randomInteger)
  validMove = True
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
    validMove = False
  
  # Do the computations only if the selected move was valid
  if validMove:  
    # Calculate the probability of response given the feature set Pi (marginal likelihood)
    marginalPi = calculateMarginalLikelihood(X, y, mu, alpha_gamma_sigma_sqr, beta_gamma_sigma_sqr, lambda_sqr[it + 1], numSamples)    
    
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
  