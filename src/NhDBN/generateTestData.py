import numpy as np
import random

def generateNetwork(num_features, independent_features, parsed_coefs, num_samples, verbose = 'false'):
  if verbose:
    print('Generating network data with:')
    print(num_features, 'features.')
    print(independent_features, 'independent feature(s).')
    print(num_samples, 'samples.\n')
  
  adjMatrix = [] # Adj matrix that will save the real config of our data
  data = np.array([])
  indep_features = []
  # Generate the independent features
  for idx in range(independent_features):
    # Generate an indep vector of normal rand
    currFeature = np.random.normal(0, 1, num_samples)
    # Append on the data
    data = np.vstack([data, currFeature]) if data.size else currFeature
    indep_features.append(idx)
    # Append info the the adj matrix
    adjMatrix.append([0 for idx in range(num_features)])

  # Transpose because the vstack generates everything transposed
  data = data.T

  # Generate a response as a func on the features
  epsilon = np.random.normal(0, 0.5, num_samples) 
  
  coefs = []
  for idx in range(num_features - independent_features):
    # Generate a vector of zeros
    currDepFeat = epsilon 

    # Select by how many indep features the feat is going to be generated
    #generated_by_num = np.random.choice(indep_features) + 1 # we +1 because the min is 0
    generated_by_num = np.random.choice([0 ,1, 2]) + 1 # New just between the fan in rest
    
    # limit the fan-in restriction
    if generated_by_num > 3:
      generated_by_num = 3
    
    # Select the indep features
    generated_by_feats = np.random.choice(indep_features, generated_by_num, replace = False) 
    # Data for the adj matrix
    currAdjMatrixInfo = [0 for idx in range(num_features)]

    coefs.append([])
    for feat in generated_by_feats:
      # Randomly specify a coef between 0 and 1 
      #currCoef = np.random.uniform(-1, 1)
      # New get the coefficients from the parsed coefs.txt file
      currCoef = parsed_coefs[idx][feat]
      # Multiply by the indep feature
      lin_comb_element = currCoef * data[:,feat]
      currDepFeat = currDepFeat + lin_comb_element
      # Track the coefs vector
      coefs[idx].append(currCoef)
      # Add to the adj matrix
      currAdjMatrixInfo[feat] = currCoef

    # Append curr Info to the adj matrix
    adjMatrix.append(currAdjMatrixInfo)
    # Generate a random number as the first data point
    noise = np.random.normal(0, 1)
    currDepFeat = np.insert(currDepFeat, 0, noise) # Append at the beggining
    currDepFeat = currDepFeat[:num_samples] # Eleminate the last one
    # Append the generated feature to the data
    data = np.append(data, currDepFeat.reshape(num_samples, 1), axis = 1)

    # For console display info
    if verbose:
      featName = independent_features + idx
      print('\nThe feature X{0} was generated {1} feature(s): '.format(featName + 1, len(generated_by_feats)))
      for feat in zip(generated_by_feats, coefs[idx]):
        print('X{0} with coefficient {1}'.format(feat[0] + 1, feat[1]))

  return data, coefs, adjMatrix
    
def generateTestDataThird(num_samples = 100, dimensions = 3):
  # Define where the changepoint is located
  change_points = [25, 50] # for now is just one segment

  data = {
    'changepoints': {}
  }
  
  # Populate with empty dicts
  for idx in range(len(change_points) + 1):
    data['changepoints'][str(idx)] = {}
    data['changepoints'][str(idx)] = {
      'features':{},
      'response':{}
    }

  # Generate independent data using the changepoings
  idxSample = 1
  for i in range(len(change_points) + 1):
    try:
      currCp = change_points[i]
    except:
      # There is no next changepoint so we set the num_samples as the uppper limit
      currCp = num_samples + 1 # TODO not sure about this + 1 at the end to fill num_samples
    if i == 0:
      lastCp = idxSample
    else:
      lastCp = change_points[i - 1]
    currSampleLen =  currCp - lastCp

    for j in range(dimensions):
      col_name = 'X' + str(j + 1)
      data['changepoints'][str(i)]['features'][col_name] = np.random.normal(0, 1, currSampleLen)

  # Generate the response Y as a func of the features
  coefs = [
    [2.8, -1.3],
    [1.8, -0.5],
    [3.4, -0,2]
  ]
  # Generate the random noise that will be added to all the samples
  epsilon = np.random.normal(0, 0.1, num_samples)   
  for i in range(len(change_points) + 1):
    # Get the cp length
    try:
      currCp = change_points[i]
    except:
      # There is no next changepoint so we set the num_samples as the uppper limit
      currCp = num_samples + 1 # TODO not sure about this + 1 at the end to fill num_samples
    if i == 0:
      lastCp = idxSample
    else:
      lastCp = change_points[i - 1]
    currSampleLen =  currCp - lastCp

    # Create the response vector y
    currCoefs = coefs[i]
    data['changepoints'][str(i)]['response']['y'] = 0.5 + \
     currCoefs[0] * data['changepoints'][str(i)]['features']['X1'][0:currSampleLen] + \
     currCoefs[1] * data['changepoints'][str(i)]['features']['X2'][0:currSampleLen] + \
     epsilon[0:currSampleLen]
  
  # Also save the unified dataset for future partitioning
  data['unified'] = {
    'features': {},
    'response': {}
  }

  for cp in data['changepoints']:
    currCpData = data['changepoints'][cp]
    # Get the current response if we have one
    try:
      accumulated = data['unified']['response']['y']
    except:
      accumulated = []
    # Concat with the current one inside the cp
    concat = np.append(accumulated, currCpData['response']['y'])
    data['unified']['response']['y'] = concat 
    
    for feature in currCpData['features']:
      # Get the current data if it exists
      try:
        accumulated = data['unified']['features'][feature]
      except:
        accumulated = []
      # Concatenate with the current one inside the cp
      concat = np.append(accumulated, currCpData['features'][feature])
      data['unified']['features'][feature] = concat
      
  return data

def generateTestDataSecond(num_samples = 100, dimensions = 3):
  data = {
    'features': {},
    'response': {}
  }

  # Generate independent data from a N(0,1)
  for i in range(dimensions):
    col_name = 'X' + str(i + 1)
    data['features'][col_name] = np.random.normal(0, 1, num_samples)

  # Generate a response as a func on the features
  epsilon = np.random.normal(0, 1, num_samples) 
  # Create the response vector y
  data['response']['y'] = 0.5 - 1.0 * data['features']['X2'] - 1.0 * data['features']['X5']  +  3 * epsilon

  return data

def testArgParse():
  generateNetwork()

def testGenerateThirdAlgo():
  print('Testing data generation...')
  dat = generateTestDataThird(100, 3)
  print('The dimensions of the response vector are: ')
  print(len(dat['changepoints']['0']['features']['X1']))
  
if __name__ == '__main__':
  testArgParse()
