import numpy as np
import random

def generateTestDataSecond(num_samples = 100, dimensions = 3):
  data = {
    'features': {},
    'response': {}
  }

  # Generate independent data form a N(0,1)
  for i in range(dimensions):
    col_name = 'X' + str(i+1)
    data['features'][col_name] = np.random.normal(0, 1, num_samples)

  # Generate a response as a func on the features
  epsilon = np.random.normal(0, 0.1, num_samples)
  # Create the response vector y
  data['response']['y'] = 0.5 + 0.8 * data['features']['X1'] - 1.3 * data['features']['X2']  + epsilon

  return data

if __name__ == '__main__':
  print('Testing data generation...')
  dat = generateTestDataSecond(100, 10)
  print('The dimensions of the response vector are: ')
  print(dat['response']['y'].shape)