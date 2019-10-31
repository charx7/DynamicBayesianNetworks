import numpy as np
import random
import pathlib

def parseCoefs(coefs_file):
  '''
    Function that parses the coefficients that are going to be used for data generation

    Args:
      coefs_file: name of the text file that contains the coefs

    Returns:
      coefs: A list with coefficients to be read
  '''
  coefs = []
  path = pathlib.Path.cwd()
  # Path handling with the debugger
  clPath = path.joinpath('src', 'NhDBN')
  # parse the coefficients file
  try:
    with open(path.joinpath(coefs_file)) as file:
      lines = file.readlines()
      for line in lines:
        # Append into the coefs list
        regularizedLine = line.strip('[]\n').replace(',', '').split(' ')
        floatifiedLine = [float(x) for x in regularizedLine] # Parse into floats
        
        coefs.append(floatifiedLine) 
  except:
    with(open(clPath.joinpath(coefs_file))) as file:
      lines = file.readlines()
      for line in lines:
        # Append into the coefs list
        regularizedLine = line.strip('[]\n').replace(',', '').split(' ')
        floatifiedLine = [float(x) for x in regularizedLine] # Parse into floats
        
        coefs.append(floatifiedLine)
  
  # Cast into floats
  return coefs

def selectData(data, featureSet):
  '''
    Selects a part of your data according to a feature set

    Args:
      data : dict of str: numpy.ndarray 
        dictionary that contains your full data to be partitioned
        featureSet: A list that contains the feature set that you want to select

    Returns:
      selectedData : dict of str: numpy.ndarray
       dictionary that contains the selected data according to the feature set
  '''
  partialData = {
    'features':{},
    'response':{}
  }

  for feature in featureSet:
    currKey = 'X' + str(int(feature))
    partialData['features'][currKey] = data['features'][currKey]
  
  return partialData

def constructMuMatrix(featureSet):
  '''
    Constructs the Mean matrix 'Mu' of the beta vector that
    depends on the parent sets plus one extra parameter for 
    the intercept  

    Args:
      featureSet : list<int>
        list of the parent sets 
    Returns:
      numpy array with the mean vector sized accordingly
  '''
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
  '''
    Constructs a design matrix with an a column of ones as the first column of the output.

    Args:
      data: A dictionary containing the features as keys.
      num_samples: The total number of data points

    Returns:
      designMatrix: A (num_samples x dimensions of your data) numpy array 
  '''

  # Construct the ones vector for the intercept
  ones_vector = np.ones(num_samples)
  
  designMatrix = ones_vector
  currFeatures = list(dict.keys(data['features']))
  # maybe the bug is here?
  currFeatures = [int(string.lstrip('X')) for string in currFeatures]
  currFeatures = sorted(currFeatures)
  currFeatures = ['X' + str(el) for el in currFeatures]

  # Stack the vectors to a giant numpy matrix
  for feature in currFeatures:
    currFeatureVector = data['features'][feature]
    designMatrix = np.vstack((designMatrix, currFeatureVector))
  
  # If the data is void then we return just the ones vector but reshaped
  if len(currFeatures) < 1:
    return  (designMatrix.T).reshape(num_samples, 1)

  # Return the transpose num_samples x features
  return designMatrix.T

def constructNdArray(data, num_samples, change_points):
  '''
    Constructs a (Design NdArray) that contains partitioned design matrices in each of its entries.

    Args:
      data (dict): data dictionary that contains a certain features and response configuration
      num_samples (int): number of samples on the data
      change_points (list<int>): list containing the changepoints on the Y idx location 
    
    Returns:
      dataNdArray (list<numpy.ndarray>): python multidimensional list that contains a numpy array that is
      a design matrix for a certain changepoint 
  '''
  # Init empty tensor
  dataNdArray = []
  # Substract 1 from the change_points because we only have num_samples - 1 points
  tmpChange_points = change_points.copy() # Operate within a copy not to mutate the list
  tmpChange_points[-1] = tmpChange_points[-1] - 1

  # Loop for each change point
  cpQueu = []
  boundCorrection = 1
  for idx, cp in enumerate(tmpChange_points):
    # Get the length of the current change point
    try:
      cpQueu.pop(0)
    except:
      cpQueu.append(1) # We are on the beginning

    # The curr len of the cp and bound lengths
    lenCurrCp = cp - cpQueu[0]
    lowerBound = cpQueu[0] 
    upperBound = cp 

    # Construct the ones vector for the intercept
    ones_vector = np.ones(lenCurrCp)
    
    currDesignMatrix = ones_vector
    currFeatures = list(dict.keys(data['features']))
    # maybe the bug is here?
    currFeatures = [int(string.lstrip('X')) for string in currFeatures]
    currFeatures = sorted(currFeatures)
    currFeatures = ['X' + str(el) for el in currFeatures]

    # Stack the vectors to a giant numpy matrix
    for feature in currFeatures:
      currFeatureVector = data['features'][feature] \
        [lowerBound - boundCorrection:cp - boundCorrection] # TODO grab data  until the changepoint
      currDesignMatrix = np.vstack((currDesignMatrix, currFeatureVector))
    
    # If the data is void then we return just the ones vector but reshaped
    if len(currFeatures) < 1:
      currDesignMatrix = ((currDesignMatrix).reshape(lenCurrCp, 1)).T # TODO handle this case

    # Transpose to get the correct result
    currDesignMatrix = currDesignMatrix.T
    # Append past cp to the Q
    cpQueu.append(cp)
    # Append to the multi dim array
    dataNdArray.append(currDesignMatrix)

  # Return the multi dim array
  return dataNdArray  

def constructResponseNdArray(y, change_points):
  '''
    Transforms the vector of response variables into an ndArray per
    segment

    Args:
      y : numpy.ndarray <int>
        a list that contains the response vectors
      change_points : list<int>
        list of changepoints 

    Returns:
      responseNdArray : list<numpy.ndarray>
        ndim list that contains a numpy array segmented according to
        the changepoints that were given
  '''
  # Init empty tensor
  responseNdArray = []
  # Substract 1 from the change_points because we only have num_samples - 1 points
  #change_points[-1] = change_points[-1] - 1

  # Loop for each change point
  cpQueu = []
  boundCorrection = 1
  for idx, cp in enumerate(change_points):
    # Get the length of the current change point
    try:
      cpQueu.pop(0)
    except:
      cpQueu.append(1) # We are on the beginning

    # The curr len of the cp and bound lengths
    lenCurrCp = cp - cpQueu[0]
    lowerBound = cpQueu[0] 
    upperBound = cp 
    # Select the appropiate data from the whole
    currResponseVector = y[lowerBound - boundCorrection:cp - boundCorrection]

    # Append the last seen cp
    cpQueu.append(cp)
    # Append to the multi dim array
    responseNdArray.append(currResponseVector)

  return responseNdArray

def deleteMove(featureSet, numFeatures, fanInRestriction, possibleFeaturesSet):
  '''
    Deletes a random feature form the set of features pi.

    Args:
        featureSet: The set of features that is going to have an element deletion.
        numFeatures: Argument to have the same args in each func type

    Returns:
        A set without a random element from the inputed featureSet.
    
    Raises:
        Exception: You cannot delete a feature when the given set just contains one element.
  '''
  if len(featureSet) < 1:
    raise ValueError('You cannot delete an element when the card(Pi) is <1.')
  
  # Randomly select one of the elements of Pi
  elToDel = np.random.choice(featureSet) # Need to set seed to change
  withoutDeleted = np.setdiff1d(featureSet, elToDel)
  return withoutDeleted

def addMove(featureSet, numFeatures, fanInRestriction, possibleFeaturesSet):
  if len(featureSet) > fanInRestriction - 1:
    raise ValueError('The cardinality of the feature set cannot be more than the fan-in restriction.')
    
  # Construct a set that contains all features idx
  #allFeatures = np.add(np.arange(numFeatures), 1) OLD use now possibleFeaturesSet

  # Construct the set where we are going to sample one feature to add randomly
  candidateFeatureSet = np.setdiff1d(possibleFeaturesSet, featureSet)
  # Select randomly an element fron the set
  featureToAdd = np.random.choice(candidateFeatureSet)
  # Append the randonly chosen feature and return
  return np.append(featureSet, featureToAdd)

def exchangeMove(featureSet, numFeatures, fanInRestriction, possibleFeaturesSet):
  if len(featureSet) < 1:
    raise ValueError('You must have at least one element on the feature set to be able to exchange it.')
  # Construct a set that contains all features idx
  #allFeatures = allFeatures = np.add(np.arange(numFeatures), 1) #OLD
  
  # Randomly select one element to exchange from Pi
  elToExchange = np.random.choice(featureSet)
  #print('The element to exchange is: ', elToExchange)
  # Remove the element from numFeatures and featureSet
  allFeaturesNoExchange = np.setdiff1d(possibleFeaturesSet, featureSet)
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
  
if __name__ == '__main__':
  #testDataGeneration()
  #testPiGeneration()
  pi = np.array([4, 5])
  featureDimensionSpace = 5
  possibleFeaturesSet = [1, 2, 3, 4, 5]
  exchangeMove(pi, featureDimensionSpace, 3, possibleFeaturesSet)
