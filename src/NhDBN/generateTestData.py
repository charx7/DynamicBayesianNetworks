import numpy as np
import random

def generateNetwork(num_features, independent_features, parsed_coefs, num_samples,  change_points, verbose = 'false'):
  if verbose:
    print('Generating network data with:')
    print(num_features, 'features.')
    print(independent_features, 'independent feature(s).')
    print(num_samples, 'samples.')
    print(change_points, 'changepoints.\n')
  
  adjMatrix = [] # Adj matrix that will save the real config of our data

  data = np.array([])
  indep_features = []
  indep_feats_adj_matrix = []
  # Generate the independent features
  for idx in range(independent_features):
    # Generate an indep vector of normal rand
    currFeature = np.random.normal(0, 1, num_samples)
    # Append on the data
    data = np.vstack([data, currFeature]) if data.size else currFeature
    indep_features.append(idx)
    # Append info the the adj matrix
    indep_feats_adj_matrix.append([0 for idx in range(num_features)])

  # Transpose because the vstack generates everything transposed
  data = data.T

  # Populate the adjMatrix for each cp
  if change_points == 0: # Handle the case where we have no changepoints
    change_points = []

  totalCps = len(change_points) + 1
  for _ in range(totalCps):
    # You have to copy otherwise it will create just x references to a single object
    nonReferenceList = indep_feats_adj_matrix.copy() 
    adjMatrix.append(nonReferenceList)

  # Generate a response as a func on the features
  epsilon = np.random.normal(0, 1, num_samples) 
  coefs = []
  change_points.append(num_samples + 2) # Append the last (artificial) change point +2 because of the bound correction 
  for idx in range(num_features - independent_features):
    # Generate a vector of zeros
    currDepFeat = epsilon # TODO consider revising this (possibly redundant feature)
    # Select by how many indep features the feat is going to be generated
    #generated_by_num = np.random.choice(indep_features) + 1 # we +1 because the min is 0
    generated_by_num = np.random.choice([0 ,1, 2]) + 1 # New just between the fan in restriction
    
    # limit the fan-in restriction
    if generated_by_num > 3:
      generated_by_num = 3
    
    # Select the indep features
    generated_by_feats = np.random.choice(indep_features, generated_by_num, replace = False) 
    
    # Loop for everychangepoint
    accCurrDepFeat = np.array([]) # Declare an empty array that will be accumulated with the cps
    cpQueu = []
    boundCorrection = 2 # Necessary due to numpy indexing
    for jdx, cp in enumerate(change_points): # We use jdx to not overwrite idx index
      # Data for the adj matrix now per CP
      currAdjMatrixInfo = [0 for idx in range(num_features)]

      # Pop an element from the cp queu
      try:
        cpQueu.pop(0)
      except:
        cpQueu.append(2) # We are on the beginning

      lowerBound = cpQueu[0] 
      upperBound = cp
      cpLen = upperBound - lowerBound + 1 

      currentChangePoint = cp # Get curr changepoint
      currDepFeat = epsilon[lowerBound - boundCorrection:currentChangePoint - boundCorrection] # take just the segment from starting noise
      coefs.append([])

      # Do the linear combination
      for feat in generated_by_feats:
        # Randomly specify a coef between 0 and 1 
        #currCoef = np.random.uniform(-1, 1)
        # New get the coefficients from the parsed coefs.txt file
        currCoef = parsed_coefs[idx + jdx][feat]
        # Multiply by the indep feature
        lin_comb_element = currCoef * data[lowerBound - boundCorrection:currentChangePoint - boundCorrection, feat]
        currDepFeat = currDepFeat + lin_comb_element
        # Track the coefs vector
        coefs[idx].append(currCoef) #TODO track coefs per cp
        # Add to the adj matrix 
        currAdjMatrixInfo[feat] = currCoef
      
      # Append to the tensor of adj matrices
      adjMatrix[jdx].append(currAdjMatrixInfo)

      # Add the current cp to the cp Stack
      cpQueu.append(cp)

      # Add an accumulator to loop between each changepoint
      accCurrDepFeat = np.append(accCurrDepFeat, currDepFeat)

    # Append curr Info to the adj matrix TODO not needed?
    #adjMatrix[idx].append(currAdjMatrixInfo)
    # Generate a random number as the first data point
    noise = np.random.normal(0, 1)
    accCurrDepFeat = np.insert(accCurrDepFeat, 0, noise) # Append at the beggining
    accCurrDepFeat = accCurrDepFeat[:num_samples] # Eliminate the last one
    # Append the generated feature to the data
    data = np.append(data, accCurrDepFeat.reshape(num_samples, 1), axis = 1)

    # For console display info TODO display info per cp
    if verbose:
      featName = independent_features + idx
      print('\nFeature X{0} was generated {1} feature(s): '.format(featName + 1, len(generated_by_feats)))
      accumCoefs = []
      for kdx in range(len(change_points)):
        print('\nOn the changepoint {0} located on {1}'.format(kdx + 1, change_points[kdx]))
        displayCoefs = adjMatrix[kdx][-1]
        for feat in generated_by_feats:
          print('X{0} with coefficient {1}'.format(feat + 1, displayCoefs[feat]))

  return data, coefs, adjMatrix #TODO remove the return of coefs (redundant since they are on adjMatrix)
    
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
