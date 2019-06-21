import numpy as np
from scipy.special import comb

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

def testFeaturesPriorProb():
  print('Executing test for calculation prior probs for a feature set...')
  dummyData = np.array([1, 3, 6])
  prob = calculateFeatureSetPriorProb(dummyData, 10, 3)
  print(prob)

if __name__ == '__main__':
  testFeaturesPriorProb()
  