import argparse
import numpy as np
from bayesianLinRegWMoves import gibbsSamplingWithMoves
from generateTestData import generateNetwork
np.random.seed(42) # Set seed for reproducibility

# Define the arg parset of the generate func
parser = argparse.ArgumentParser(description = 'Specify the type of data to be generated.')
parser.add_argument('-n_f', '--num_features', metavar='', type = int, default = 3,
  help = 'Number of features to be generated on the network.')
parser.add_argument('-n_s', '--num_samples', metavar='', type = int, default = 100,
  help = 'Number of data points that are going to be generated.')
parser.add_argument('-n_i', '--num_indep', metavar='', type = int, default = 1,
  help = 'Number of independent features.')
# Mutually exclusive arguments
group  = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true', help = 'Print verbose.')
args = parser.parse_args()

def testBayesianLinRegWithMoves():
  print('Testing Bayesian Lin Reg with moves.')
  # Generate data to test our algo
  network, coefs = generateNetwork(args.num_features, args.num_indep, args.num_samples, args.verbose)
  
  # Get the dimensions of the data
  dims = network.shape[1]
  dimsVector = [x for x in range(dims)]

  # Set the ith as the response the rest as features
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
      data['features'][col_name] = network[:,el]
    # Add the response to the dict
    data['response']['y'] = network[:, currResponse]
  
    # Do the gibbs Sampling
    results = gibbsSamplingWithMoves(data, args.num_samples)
    
    
def main():
  if args.verbose:
    print('Generating network data with:')
    print(args.num_features, 'features.')
    print(args.num_indep, 'independent feature(s).')
    print(args.num_samples, 'samples.\n')

  testBayesianLinRegWithMoves()

if __name__ == "__main__":
  main()