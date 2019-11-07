import numpy as np
import random
import logging
from systemUtils import writeOutputFile

# Logger configuration TODO move this into a config file
logger = logging.getLogger(__name__) # create a logger obj
logger.setLevel(logging.INFO) # establish logging level
# Establish the display of the logger
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s') 
file_handler = logging.FileHandler('output.log', mode='a') # The file output name
# Add the formatter to the logger
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def generateNetwork(num_features, independent_features, parsed_coefs, num_samples,
  change_points, verbose = 'false', generated_noise_var = 1):
  '''
    Documentation is missing for the function
  '''
  if verbose:
    # Output file write
    output_line =  (
      'Generating a network data with\n' +
      str(num_features) + ' features.\n' +
      str(independent_features) + ' independent feature(s).\n' +
      str(num_samples) + ' samples.\n' +
      str(generated_noise_var) + ' agregated noise on the dependent features.\n' +
      str(change_points) + ' changepoints.\n')
    print(output_line) ; logger.info(output_line) # write output str to the file

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
  epsilon = np.random.normal(0, generated_noise_var, num_samples) 
  coefs = []
  change_points.append(num_samples + 1) # Append the last (artificial) change point +2 because of the bound correction 
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
    boundCorrection = 1 # Necessary due to numpy indexing
    for jdx, cp in enumerate(change_points): # We use jdx to not overwrite idx index
      # Data for the adj matrix now per CP
      currAdjMatrixInfo = [0 for idx in range(num_features)]

      # Pop an element from the cp queu
      try:
        cpQueu.pop(0)
      except:
        cpQueu.append(1) # We are on the beginning

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
      # print to the console and save to the output file
      output_line = (
        'Feature X{0} was generated using {1} feature(s): '.format(featName + 1, len(generated_by_feats))
      )
      print(output_line) ; logger.info(output_line)
      # Print for each changepoint
      accumCoefs = []
      for kdx in range(len(change_points)):
        # print and write
        output_line = (
          '\nOn the changepoint {0} located on {1}\n'.format(kdx + 1, change_points[kdx])
        )
        print(output_line) ; logger.info(output_line)
        
        displayCoefs = adjMatrix[kdx][-1]
        for feat in generated_by_feats:
          # print and write
          output_line = ('X{0} with coefficient {1}\n'.format(feat + 1, displayCoefs[feat]))
          print(output_line) ; logger.info(output_line)

  return data, coefs, adjMatrix #TODO remove the return of coefs (redundant since they are on adjMatrix)
    