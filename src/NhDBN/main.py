import argparse
import numpy as np
from bayesianLinRegWMoves import gibbsSamplingWithMoves
from generateTestData import generateNetwork
from utils import parseCoefs
from scores import calculateFeatureScores, drawRoc
from pprint import pprint

np.random.seed(41) # Set seed for reproducibility

# Define the arg parset of the generate func
parser = argparse.ArgumentParser(description = 'Specify the type of data to be generated.')
parser.add_argument('-n_f', '--num_features', metavar='', type = int, default = 3,
  help = 'Number of features to be generated on the network.')
parser.add_argument('-n_s', '--num_samples', metavar='', type = int, default = 100,
  help = 'Number of data points that are going to be generated.')
parser.add_argument('-n_i', '--num_indep', metavar='', type = int, default = 1,
  help = 'Number of independent features.')
parser.add_argument('-c_f', '--coefs_file', metavar='', type = str, default='coefs.txt',
  help = 'filename of the coefficients for the network data generation.')
parser.add_argument('-c_l', '--chain_length', metavar='', type = int, default = 5000,
  help = 'amount of iterations for the MCMC algorithm.')
parser.add_argument('-b_i', '--burn_in', metavar='', type= int, default = 1000,
  help = 'burn in period for the MCMC chain.')
parser.add_argument('-c_p', '--change_points', metavar='', type = int, default = 0, nargs='+',
  help = 'a series of change points that will be generated. ')
# Mutually exclusive arguments
group  = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true', help = 'Print verbose.')
args = parser.parse_args()

def testBayesianLinRegWithMoves(coefs):
  print('Testing Bayesian Lin Reg with moves.')
  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep, coefs, args.num_samples,
  args.change_points, args.verbose)
  # Since we are testing with no cps get the single adjMatrix
  adjMatrix = adjMatrix[0]
  # Get the dimensions of the data
  dims = network.shape[1]
  dimsVector = [x for x in range(dims)]

  # Set the ith as the response the rest as features
  proposedAdjMatrix = [] # Proposed adj matrix that will be populated by the algo (edge score matrix)
  for configuration in dimsVector:
    data = {
      'features': {},
      'response': {}
    }

    currResponse = configuration
    # You have to evaluate because the filter returns an obj
    currFeatures = list(filter(lambda x: x != configuration, dimsVector))
    
    # Add the features to the dict
    for el in currFeatures:
      col_name = 'X' + str(el)
      data['features'][col_name] = network[:args.num_samples - 1,el]

    # Add the response to the dict
    data['response']['y'] = network[1:, currResponse]
  
    # Do the gibbs Sampling
    results = gibbsSamplingWithMoves(data, args.num_samples - 1, args.chain_length)
    res = calculateFeatureScores(results['pi_vector'][:args.burn_in], dims, currFeatures, currResponse)
    proposedAdjMatrix.append(res)

  # Return the proposed adj matrix
  return proposedAdjMatrix, adjMatrix

def testNoCps():
  # The coefficients that will be used to generate the random data
  coefs = parseCoefs(args.coefs_file)
  adjMatrixProp, trueAdjMatrix = testBayesianLinRegWithMoves(coefs)
  
  if args.verbose:
    print('\nThe true adj matrix is: \n')
    pprint(trueAdjMatrix)
    print('\nThe proposed adj matrix is: \n')
    pprint(adjMatrixProp)
  # Remove the diagonal that is allways going to be right
  trueAdjMatrixNoDiag = []
  idxToRemove = 0
  for row in trueAdjMatrix:
    row.pop(idxToRemove)
    trueAdjMatrixNoDiag.append(row)
    idxToRemove + 1
  # Now for the inferred matrix  
  adjMatrixPropNoDiag = []
  idxToRemove = 0
  for row in adjMatrixProp:
    row.pop(idxToRemove)
    adjMatrixPropNoDiag.append(row)
    idxToRemove + 1
  # Re-assign them
  trueAdjMatrix = trueAdjMatrixNoDiag
  adjMatrixProp = adjMatrixPropNoDiag

  # Flatten the adj matrix to pass to the RoC
  flattened_true = [item for sublist in trueAdjMatrix for item in sublist]
  flattened_true = [1 if item else 0 for item in flattened_true] # convert to binary response vector
  flattened_scores = [item for sublist in adjMatrixProp for item in sublist]
  
  drawRoc(flattened_scores, flattened_true) # Draw the RoC curve

def testTestPwBlrWMoves():
  # The coefficients that will be used to generate the random data
  coefs = parseCoefs(args.coefs_file)
  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep, coefs, args.num_samples,
   args.change_points, args.verbose)
  
def main():
  testNoCps() # Uncomment for testing the second algo on a network
  #testTestPwBlrWMoves() # Uncomment to test the third algo on a network

if __name__ == "__main__":
  main()