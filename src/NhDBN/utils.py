import numpy as np
import random

def constructMuMatrix(featureSet):
  # Set the number of vars in the featureSet
  numFeatures = len(featureSet) + 1 # +1 because of the intercept?
  # Prior expectation is the zero vector
  return(np.zeros(numFeatures).reshape(numFeatures, 1)) 
  
def generateInitialFeatureSet(numFeatures, fanInRestriction):
  '''
    Function that initializes the feature set Pi.

    Args:
        numFeatures: Total number of features of the data.
        fanInRestriction: Maximum size of the set Pi.

    Returns:
        A numpy array that cointains a random set of initial features.
  '''
  randomIdx = np.random.choice(numFeatures, fanInRestriction, replace=False)
  randomIdx = np.add(randomIdx, 1)
  return randomIdx

def generateData(num_samples = 100, dimensions = 3, dependent = 1):
  #np.random.seed(42)

  data = {
    'features': {},
    'response': {}
  }

  # Generate independent data form a N(0,1)
  for i in range(dimensions - dependent):
    col_name = 'X' + str(i+1)
    data['features'][col_name] = np.random.normal(0, 1, num_samples)

  # Generate the rest of the variables as a linear combination of the indep ones
  for i in range(dependent):
    # Get the data name for the dependent columns
    currDataIdx = (dimensions - dependent) + i + 1
    col_name = 'X' + str(currDataIdx)

    # The random noise that will be added    
    epsilon = np.random.normal(0, 0.1, num_samples)
    currData = (data['features']['X1'] * -0.2 + data['features']['X2'] * 0.7) + epsilon

    data['features'][col_name] = currData

  # Create the response vector y
  data['response']['y'] = np.random.normal(0, 1, num_samples)

  return data

def constructDesignMatrix(data, num_samples):
  # Construct the ones vector for the intercept
  ones_vector = np.ones(num_samples)
  
  designMatrix = ones_vector
  currFeatures = list(dict.keys(data['features']))
  # Stack the vectors to a giant numpy matrix
  for feature in currFeatures:
    currFeatureVector = data['features'][feature]
    designMatrix = np.vstack((designMatrix, currFeatureVector))
  
  # Return the transpose num_samples x features
  return designMatrix.T

def deleteMove(featureSet):
  '''
    Deletes a random feature form the set of features pi.

    Args:
        featureSet: The set of features that is going to have an element deletion.

    Returns:
        A set without a random element from the inputed featureSet.

    Raises:
        Exception: You cannot delete a feature when the given set just contains one element.
  '''
  if len(featureSet) < 2:
    raise ValueError('You cannot delete an element when the card(Pi) is <2.')
  
  # Randomly select one of the elements of Pi
  elToDel = np.random.choice(featureSet) # Need to set seed to change
  withoutDeleted = np.setdiff1d(featureSet, elToDel)
  return withoutDeleted

def addMove(featureSet, numFeatures):
  # Construct a set that contains all features idx
  allFeatures = np.add(np.arange(numFeatures), 1)
  # Construct the set where we are going to sample one feature to add randomly
  candidateFeatureSet = np.setdiff1d(allFeatures, featureSet)
  # Select randomly an element fron the set
  featureToAdd = np.random.choice(candidateFeatureSet)
  # Append the randonly chosen feature and return
  return np.append(featureSet, featureToAdd)

def exchangeMove(featureSet, numFeatures):
  # Construct a set that contains all features idx
  allFeatures = allFeatures = np.add(np.arange(numFeatures), 1)
  # Randomly select one element to exchange from Pi
  elToExchange = np.random.choice(featureSet)
  #print('The element to exchange is: ', elToExchange)
  # Remove the element from numFeatures and featureSet
  allFeaturesNoExchange = np.setdiff1d(allFeatures, elToExchange)
  featureSet = np.setdiff1d(featureSet, elToExchange)
  # Select randomly a element to add to the feature set
  elToAdd = np.random.choice(allFeaturesNoExchange)
  #print('The element to add is: ', elToAdd)
  # Add the element to the featureSet
  return np.append(featureSet, elToAdd)

def testDataGeneration():
  print('Executing test...')
  # Basic test
  data = generateData()
  print(len(data['features']))
  X = constructDesignMatrix(data, 100)
  print(X.shape)

def testPiGeneration():
  print('Executing pi test generation...')
  rndSet = generateInitialFeatureSet(6, 3)
  print(rndSet)

  # Test delete Move
  #print(deleteMove(rndSet))
  # Test add Move
  #print(addMove(rndSet, 10))
  # Test element Move
  #print(exchangeMove(rndSet, 10))

if __name__ == '__main__':
  testDataGeneration()
  testPiGeneration()
