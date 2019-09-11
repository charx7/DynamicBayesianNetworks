import argparse
import numpy as np
from bayesianLinRegWMoves import gibbsSamplingWithMoves
from pWLinRegNhdbn import pwGibbsSamplingWithMoves, pwGibbsSamplingWithCpsParentsMoves
from generateTestData import generateNetwork
from utils import parseCoefs
from scores import calculateFeatureScores, adjMatrixRoc
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
parser.add_argument('-g_n_v', '--generated_noise_var', metavar='', type = float, default = 1,
  help = 'the variance of the noise that is going to generate the dependent features')
# Mutually exclusive arguments
group  = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true', help = 'Print verbose.')
args = parser.parse_args()
  
def testBayesianLinRegWithMoves(coefs):
  print('Testing Bayesian Lin Reg with moves.')
  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep, coefs, args.num_samples,
  args.change_points, args.verbose, args.generated_noise_var)
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
  adjMatrixRoc(adjMatrixProp, trueAdjMatrix, args.verbose)
  
def testTestPwBlrWMoves():
  # The coefficients that will be used to generate the random data
  coefs = parseCoefs(args.coefs_file)
  print('Testing Piece-Wise Bayesian Lin Reg with moves.')
  
  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep, coefs, args.num_samples,
  args.change_points, args.verbose, args.generated_noise_var)
  
  # Get the dimensions of the data
  dims = network.shape[1]
  dimsVector = [x for x in range(dims)]

  # Set the ith as the response the rest as features
  proposedAdjMatrix = [] # TODO multi dim... Proposed adj matrix that will be populated by the algo (edge score matrix)
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
      data['features'][col_name] = network[:args.num_samples - 1, el]

    # Add the response to the dict
    data['response']['y'] = network[1:, currResponse]
  
    # Do the gibbs Sampling
    results = pwGibbsSamplingWithMoves(data, args.change_points, args.num_samples - 1, args.chain_length)
    # Calculate feature Scores  
    res = calculateFeatureScores(results['pi_vector'][:args.burn_in], dims, currFeatures, currResponse)
    proposedAdjMatrix.append(res)

  # Return the proposed adj matrix
  #return proposedAdjMatrix, adjMatrix # uncomment to functionalize
  trueAdjMatrix = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  adjMatrixRoc(proposedAdjMatrix, trueAdjMatrix, args.verbose)
  
def testPwBlrWithCpsParentsMoves():
  # The coefficients that will be used to generate the random data
  coefs = parseCoefs(args.coefs_file)
  print('Testing Piece-Wise Bayesian Lin Reg with moves on Cps and Parent sets.')
  
  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep, coefs, args.num_samples,
  args.change_points, args.verbose, args.generated_noise_var)
  
  # Get the dimensions of the data
  dims = network.shape[1]
  dimsVector = [x for x in range(dims)]

  # Set the ith as the response the rest as features
  proposedAdjMatrix = [] # TODO multi dim... Proposed adj matrix that will be populated by the algo (edge score matrix)
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
      data['features'][col_name] = network[:args.num_samples - 1, el]

    # Add the response to the dict
    data['response']['y'] = network[1:, currResponse]
  
    # Do the gibbs Sampling
    results = pwGibbsSamplingWithCpsParentsMoves(data, args.change_points,
     args.num_samples - 1, args.chain_length)

    # Calculate feature Scores  
    res = calculateFeatureScores(results['pi_vector'][:args.burn_in], dims, currFeatures, currResponse)
    proposedAdjMatrix.append(res)

  # Return the proposed adj matrix
  #return proposedAdjMatrix, adjMatrix # uncomment to functionalize
  trueAdjMatrix = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  adjMatrixRoc(proposedAdjMatrix, trueAdjMatrix, args.verbose)

def main():
  #testNoCps() # Uncomment for testing the second algo on a network
  #testTestPwBlrWMoves() # Uncomment to test the third algo on a network
  testPwBlrWithCpsParentsMoves()

if __name__ == "__main__":
  main()