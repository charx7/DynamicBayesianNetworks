import argparse
import logging
import numpy as np
from utils import transformResults, adjMatrixRoc
from dyban.generateTestData import generateNetwork
from dyban.utils import parseCoefs
from dyban.systemUtils import cleanOutput, writeOutputFile
from dyban import Network
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
parser.add_argument('-m', '--method', metavar='', type = str, default = 'h-dbn',
  help = 'what method will be run')
parser.add_argument('-l', '--lag', metavar='', type = int, default = 1,
  help = 'lag of the time series')

# Mutually exclusive arguments
group  = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true', help = 'Print verbose.')
args = parser.parse_args()

# logging cofiguration
logger = logging.getLogger(__name__) # create a logger obj
logger.setLevel(logging.INFO) # establish logging level
# Establish the display of the logger
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s') 
file_handler = logging.FileHandler('output.log', mode='w') # The file output name
# Add the formatter to the logger
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def test_h_dbn(coefs):
  output_line = (
    'Bayesian Linear Regression with moves on' +
    'the parent set only. \n'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  change_points = [] # set the cps empty list because this is the homegeneous version
  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep,
  coefs, args.num_samples, change_points, args.verbose, args.generated_noise_var)

  baNet = Network([network], args.chain_length, args.burn_in, args.lag, change_points) # Create theh BN obj
  baNet.infer_network('h_dbn') # Do the fixed parents version of the DBN algo
  
  true_inc = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  flattened_true, flattened_scores = transformResults(true_inc, baNet.proposed_adj_matrix)
  adjMatrixRoc(flattened_scores, flattened_true, args.verbose)

def testPwBlrWithParentMoves(coefs):
  output_line = (
    'Bayesian Piece-Wise Linear Regression with moves on' +
    'the parent set only with fixed changepoints. \n'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep, coefs, args.num_samples,
  args.change_points, args.verbose, args.generated_noise_var)

  baNet = Network(network, args.chain_length, args.burn_in, args.change_points) # Create theh BN obj
  baNet.infer_network('fixed_nh_dbn') # Do the fixed chnagepoints version of the DBN algo

  true_inc = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  flattened_true, flattened_scores = transformResults(true_inc, baNet.proposed_adj_matrix)
  adjMatrixRoc(flattened_scores, flattened_true, args.verbose)

def testPwBlrWithCpsParentMoves(coefs):
  output_line = (
    'Bayesian Piece-Wise Linear Regression with moves on' +
    'change-points and parent sets.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep,
  coefs, args.num_samples, args.change_points, args.verbose, args.generated_noise_var)

  baNet = Network([network], args.chain_length, args.burn_in, args.lag)
  baNet.infer_network('varying_nh_dbn')

  trueAdjMatrix = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  flattened_true, flattened_scores = transformResults(trueAdjMatrix, baNet.proposed_adj_matrix)
  adjMatrixRoc(flattened_scores, flattened_true, args.verbose)

def testSeqCoupPwBlrWithCpsParentMoves(coefs):
  output_line = (
    'Sequentially Coupled Bayesian Piece-Wise Linear Regression with moves on' +
    'change-points and parent sets.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep,
  coefs, args.num_samples, args.change_points, args.verbose, args.generated_noise_var)

  baNet = Network([network], args.chain_length, args.burn_in, args.lag)
  baNet.infer_network('seq_coup_nh_dbn')

  trueAdjMatrix = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  adjMatrixRoc(baNet.proposed_adj_matrix, trueAdjMatrix, args.verbose)

def testGlobCoupPwBlrWithCpsParentMoves(coefs):
  output_line = (
    'Globally Coupled Bayesian Piece-Wise Linear Regression with moves on' +
    'change-points and parent sets.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  # Generate data to test our algo
  network, _, adjMatrix = generateNetwork(args.num_features, args.num_indep,
  coefs, args.num_samples, args.change_points, args.verbose, args.generated_noise_var)

  baNet = Network([network], args.chain_length, args.burn_in, args.lag)
  baNet.infer_network('glob_coup_nh_dbn')

  trueAdjMatrix = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  adjMatrixRoc(baNet.proposed_adj_matrix, trueAdjMatrix, args.verbose)

def main():
  # The coefficients that will be used to generate the random data
  coefs = parseCoefs(args.coefs_file)
  # Select and run the chosen algorithm
  if args.method == 'h-dbn':
    test_h_dbn(coefs) # Uncomment for testing the second algo on a network  
  elif args.method == 'nh-dbn':
    testPwBlrWithCpsParentMoves(coefs) # Test the fourth algorithm  
  elif args.method == 'seq-dbn':
    testSeqCoupPwBlrWithCpsParentMoves(coefs) # test the fifth algorithm  
  elif args.method == 'glob-dbn':
    testGlobCoupPwBlrWithCpsParentMoves(coefs) # test the sixth algorithm
  #elif args.method == 'var-glob-dbn':
  #  testVvGlobCoup(data, true_inc)

if __name__ == "__main__":
  main()
  