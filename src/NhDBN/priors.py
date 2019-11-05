import numpy as np
from scipy.special import comb

def calculateChangePointsSetPrior(changepointsSet):
  '''
    Function that returns the prior probability of the changepoint set tau

    Args:
      changepointsSet : list<int>
        a list that contains the proposed changepoint set
    
    Returns: 
      res : float
        the prior probability of the changepointset
  '''
  
  changepointsSetCopy = changepointsSet.copy() # make a copy because of mutability
  changepointsSetCopy.insert(0, 1) # insert the t0 value

  p = 0.125 # hyperparameter of the prior distribution of the changepoints
  
  # The changepoints set cannot be larger than 9 elements
  if len(changepointsSetCopy) > 10:  # consider the first pseudo-changepoint
    res = 0
  else: # Do the computation of the prior
    # TODO check if this is valid for empty changepointsSet(s)
    # Get the last element of the changepointsSet
    tau_h = changepointsSetCopy[-1] - 1 # (-2) because of the definition of tau_h and N
    tau_h_minus = changepointsSetCopy[-2] # Get the second to last element
    
    el1 = (1 - p) ** (tau_h - tau_h_minus) # first part of the computation

    # Loop for the product
    el2 = 1
    for idx in range(len(changepointsSet) - 1):
      # Get the taus location
      curTau_h = changepointsSetCopy[idx + 1]
      curTau_h_minus = changepointsSetCopy[idx]
      # Do the product
      currEl2 = p * (1 - p) ** (curTau_h - curTau_h_minus - 1)

      el2 = el2 * currEl2
    
    res = el1 * el2
    
  return res

def calculateFeatureSetPriorProb(featureSet, featuresDimensions, fanInRestriction):
  '''
    Function that calculate the prior probability of the feature set Pi.

    Args:
        featureSet: A numpy array of the set of variables on Pi.
        featuresDimensions: Integer containing the dimension of the set of features
        fanInRestriction: Maximum size of the set Pi.

    Returns:
        The prior probability of the set Pi.
  '''
  if len(featureSet) > 3:
    return 0

  c = 0
  for i in range(fanInRestriction + 1):
    currC = comb(featuresDimensions, i)
    c = c + currC
  
  return (1 / c) 
