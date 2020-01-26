import argparse
import logging
import numpy as np
from generateTestData import generateNetwork
from systemUtils import cleanOutput, writeOutputFile, data_reader
from utils import save_chain
from scores import calculateFeatureScores, adjMatrixRoc, transformResults
from network import Network
from pprint import pprint

#np.random.seed(41) # Set seed for reproducibility

# Define the arg parset of the generate func
parser = argparse.ArgumentParser(description = 'Specify the type of data to be generated.')
parser.add_argument('-c_l', '--chain_length', metavar='', type = int, default = 5000,
  help = 'amount of iterations for the MCMC algorithm.')
parser.add_argument('-b_i', '--burn_in', metavar='', type= int, default = 1000,
  help = 'burn in period for the MCMC chain.')
parser.add_argument('-c_p', '--change_points', metavar='', type = int, default = 0, nargs='+',
  help = 'a series of change points that will be generated. ')
parser.add_argument('-m', '--method', metavar='', type = str, default = 'h-dbn',
  help = 'what method will be run')

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

def read_yeast():
  # Read the yeast data
  dataOff = data_reader('./data/datayeastoff.txt')
  dataOff = np.delete(dataOff, 0, 0) # delete the first row (misread headers)
  dataOn = data_reader('./data/datayeaston.txt')
  dataOn = np.delete(dataOn, 0, 0) # delte the first row (misread headers)

  merged_data = np.vstack((dataOn, dataOff)) # merge the on + off datasets

  # Set the true incidence matrix defined by the literature
  # true_inc = [
  #   [0, 1, 0, 0, 0],
  #   [0, 0, 1, 1, 0],
  #   [1, 0, 0, 1 ,1],
  #   [0, 1, 0, 0, 0],
  #   [1, 0, 0, 0, 0]
  # ]
  true_inc = [
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0 ,0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0]
  ]
  return(merged_data, true_inc)

def testGlobCoupPwBlrWithCpsParentMoves(data, true_inc):
  output_line = (
    'Globally Coupled Bayesian Piece-Wise Linear Regression with moves on ' +
    'change-points and parent sets on Yeast Data.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  baNet = Network(data, args.chain_length, args.burn_in)
  baNet.infer_network('glob_coup_nh_dbn')

  adjMatrixRoc(baNet.proposed_adj_matrix, true_inc, args.verbose)
  
  # save the chain into the output folder
  save_chain('glob_coup_dbn.pckl', baNet)

def testSeqCoupPwBlrWithCpsParentMoves(data, true_inc):
  output_line = (
    'Sequentially Coupled Bayesian Piece-Wise Linear Regression with moves on ' +
    'change-points and parent sets on Yeast data.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  baNet = Network(data, args.chain_length, args.burn_in)
  baNet.infer_network('seq_coup_nh_dbn')

  adjMatrixRoc(baNet.proposed_adj_matrix, true_inc, args.verbose)

  # save the chain into the output folder
  save_chain('seq_coup_dbn.pckl', baNet)

def testPwBlrWithCpsParentMoves(data, true_inc):
  output_line = (
    'Bayesian Piece-Wise Linear Regression with moves on ' +
    'change-points and parent sets for the Yeast data.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output
  if args.change_points == 0:
    args.change_points = []
  args.change_points.append(data.shape[0] + 1) # append the len data + 1 so the algo works
  baNet = Network(data, args.chain_length, args.burn_in)
  baNet.infer_network('varying_nh_dbn')

  adjMatrixRoc(baNet.proposed_adj_matrix, true_inc, args.verbose)

  # save the chain into the output folder
  save_chain('nh_dbn.pckl', baNet)

def testPwBlrWithParentMoves(data, true_inc):
  output_line = (
    'Bayesian Piece-Wise Linear Regression with moves on ' +
    'the parent set only with fixed changepoints for the Yeast Data. \n'
    )
  print(output_line) ; logger.info(output_line) # Print and write output
  if args.change_points == 0:
    args.change_points = []
  args.change_points.append(data.shape[0] + 1) # append the len data + 1 so the algo works
  baNet = Network(data, args.chain_length, args.burn_in, args.change_points) # Create theh BN obj
  baNet.infer_network('fixed_nh_dbn') # Do the fixed changepoints version of the DBN algo

  adjMatrixRoc(baNet.proposed_adj_matrix, true_inc, args.verbose) # check the ROC

def test_h_dbn(data, true_inc):
  output_line = (
    'Bayesian Linear Regression with moves on ' +
    'the parent set only for the Yeast data. \n'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  change_points = [] # set the cps empty list because this is the homegeneous version

  # Create/Call the Network objects/methods
  baNet = Network(data, args.chain_length, args.burn_in, args.change_points) # Create theh BN obj
  baNet.infer_network('h_dbn') # Do the fixed parents version of the DBN algo
  
  # trueAdjMatrix = adjMatrix[0] # For the moment we just get the adj matrix of the first cp
  adjMatrixRoc(baNet.proposed_adj_matrix, true_inc, args.verbose)

  # save the chain into the output folder
  save_chain('h_dbn.pckl', baNet)

def main():
  #cleanOutput() # clean output folder #TODO make it an argumento to clean the folder
  data, true_inc = read_yeast() # read the YEAST data
  
  # Select and run the chosen algorithm
  if args.method == 'h-dbn':
    test_h_dbn(data, true_inc)
    #testPwBlrWithParentMoves(data, true_inc) # this one is for debug
  elif args.method == 'nh-dbn':
    testPwBlrWithCpsParentMoves(data, true_inc)
  elif args.method == 'seq-dbn':
    testSeqCoupPwBlrWithCpsParentMoves(data, true_inc)
  elif args.method == 'glob-dbn':
    testGlobCoupPwBlrWithCpsParentMoves(data, true_inc)

if __name__ == "__main__":
  main()
  