import numpy as np

def generateData(num_samples = 100, dimensions = 3, dependent = 1):
  np.random.seed(42)

  data = {
    'features': {},
    'response': {}
  }

  # Generate independent data form a N(0,1)
  for i in range(dimensions - dependent):
    col_name = 'X' + str(i+1)
    data['features'][col_name] = np.random.normal(0, 1, num_samples)

  # Generate the rest of the variables as a linear combination of the indep ones
  for i in range(dependent):
    # Get the data name for the dependent columns
    currDataIdx = (dimensions - dependent) + i + 1
    col_name = 'X' + str(currDataIdx)

    # The random noise that will be added    
    epsilon = np.random.normal(0, 0.1, num_samples)
    currData = (data['features']['X1'] * -0.2 + data['features']['X2'] * 0.7) + epsilon

    data['features'][col_name] = currData

  # Create the response vector y
  data['response']['y'] = np.random.normal(0, 1, num_samples)

  return data

def constructDesignMatrix(data, num_samples):
  # Construct the ones vector for the intercept
  ones_vector = np.ones(num_samples)
  
  designMatrix = ones_vector
  numOfFeatures = len(data['features'])
  # Stack the vectors to a giant numpy matrix
  for i in range(numOfFeatures):
    currFeatureVector = data['features']['X' + str(i+1)]
    designMatrix = np.vstack((designMatrix, currFeatureVector))
  
  # Return the transpose num_samples x features
  return designMatrix.T
  
if __name__ == '__main__':
  print('Executing test...')
  # Basic test
  data = generateData()
  print(len(data['features']))
  X = constructDesignMatrix(data, 100)
  print(X.shape)
  