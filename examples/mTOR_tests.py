import argparse
import logging
import numpy as np
import pandas as pd
from dyban import Network
from utils import  adjMatrixRoc, transformResults, save_chain

# Define the arg parset of the generate func
parser = argparse.ArgumentParser(description = 'Specify the hyper-params of the method to be computed.')
parser.add_argument('-c_l', '--chain_length', metavar='', type = int, default = 5000,
  help = 'amount of iterations for the MCMC algorithm.')
parser.add_argument('-b_i', '--burn_in', metavar='', type = int, default = 1000,
  help = 'burn in period for the MCMC chain.')
parser.add_argument('-c_p', '--change_points', metavar='', type = int, default = 0, nargs='+',
  help = 'a series of change points that will be generated. ')
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

def read_mTOR():
  # Read the yeast data
  data = pd.read_csv('./examples/data/gp_mTOR.csv')
  data = data.to_numpy()
  return [data]

def testVvGlobCoup(data):
  output_line = (
    'Varying Variances Globally Coupled Bayesian Piece-Wise Linear Regression with moves on ' +
    'change-points and parent sets on mTOR Data.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  baNet = Network(data, args.chain_length, args.burn_in, args.lag)
  baNet.infer_network('var_glob_coup_nh_dbn')

  # save the chain into the output folder
  save_chain('mTOR_vv_glob_coup_dbn.pckl', baNet)
  return baNet.proposed_adj_matrix

def testGlobCoupPwBlrWithCpsParentMoves(data):
  output_line = (
    'Globally Coupled Bayesian Piece-Wise Linear Regression with moves on ' +
    'change-points and parent sets on mTOR Data.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  baNet = Network(data, args.chain_length, args.burn_in, args.lag)
  baNet.infer_network('glob_coup_nh_dbn')

  # save the chain into the output folder
  save_chain('mTOR_glob_coup_dbn.pckl', baNet)
  return baNet.proposed_adj_matrix
  
def testSeqCoupPwBlrWithCpsParentMoves(data):
  output_line = (
    'Sequentially Coupled Bayesian Piece-Wise Linear Regression with moves on ' +
    'change-points and parent sets on mTOR data.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  baNet = Network(data, args.chain_length, args.burn_in, args.lag)
  baNet.infer_network('seq_coup_nh_dbn')

  # save the chain into the output folder
  save_chain('mTOR_seq_coup_dbn.pckl', baNet)

  return baNet.proposed_adj_matrix
  
def testPwBlrWithCpsParentMoves(data):
  output_line = (
    'Bayesian Piece-Wise Linear Regression with moves on ' +
    'change-points and parent sets for the mTOR data.'
  )
  print(output_line) ; logger.info(output_line) # Print and write output
  if args.change_points == 0:
    args.change_points = []
  
  # to get the total length of the data over the segments
  # data_len = 0
  # for segment in data:
  #   data_len = data_len + segment.shape[0]

  # args.change_points.append(data_len + 1) # append the len data + 1 so the algo works
  baNet = Network(data, args.chain_length, args.burn_in, args.lag)
  baNet.infer_network('varying_nh_dbn')

  # save the chain into the output folder
  save_chain('mTOR_nh_dbn.pckl', baNet)

  return baNet.proposed_adj_matrix

def test_h_dbn(data):
  output_line = (
    'Bayesian Linear Regression with moves on ' +
    'the parent set only for the mTOR data. \n'
  )
  print(output_line) ; logger.info(output_line) # Print and write output

  change_points = [] # set the cps empty list because this is the homegeneous version

  # Create/Call the Network objects/methods
  baNet = Network(data, args.chain_length, args.burn_in, args.lag, args.change_points) # Create theh BN obj
  baNet.infer_network('h_dbn') # Do the fixed parents version of the DBN algo
  
  # save the chain into the output folder
  save_chain('mTOR_h_dbn.pckl', baNet)

  return baNet.proposed_adj_matrix
  
def main():
  data = read_mTOR() # read the YEAST data
  
  # Select and run the chosen algorithm
  if args.method == 'h-dbn':
    res = test_h_dbn(data)
    #testPwBlrWithParentMoves(data, true_inc) # this one is for debug
  elif args.method == 'nh-dbn':#save_chain('glob_coup_dbn.pckl', baNet)

    res = testPwBlrWithCpsParentMoves(data)
  elif args.method == 'seq-dbn':
    res = testSeqCoupPwBlrWithCpsParentMoves(data)
  elif args.method == 'glob-dbn':
    res = testGlobCoupPwBlrWithCpsParentMoves(data)
  elif args.method == 'var-glob-dbn':
    res = testVvGlobCoup(data)

  adj_matrix = np.array(res).T # transpose because the results in baNet are transposed
  
  # strong signals >0.8
  adj_matrix_strong = np.where(adj_matrix > 0.8, adj_matrix, 0)

  proteins = [
      'IR-beta-pY1146',
      'IRS-pS636/639',
      'AMPK-pT172',
      'TSC2-pS1387',
      'Akt-pT308',
      'Akt-pS473',
      'mTOR-pS2448',
      'mTOR-pS2481',
      'p70-S6K-pT389',
      'PRAS40-pS183',
      'PRAS40-pT246'
  ]

  for idx, protein in enumerate(proteins):
    curr_row = adj_matrix_strong[idx,:] 
    for jdx, col in enumerate(curr_row):
      if col > 0: 
        print(protein, ' -> ', proteins[jdx]) 
      
if __name__ == "__main__":
  main()
  