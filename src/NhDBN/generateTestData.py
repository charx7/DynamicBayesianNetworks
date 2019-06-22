import numpy as np
import random

def generateTestDataThird(num_samples = 100, dimensions = 3):
  # Define where the changepoint is located
  change_points = [25, 50] # for now is just one segment

  data = {
    'changepoints': {}
  }
  
  # Populate with empty dicts
  for idx in range(len(change_points) + 1):
    data['changepoints'][str(idx)] = {}
    data['changepoints'][str(idx)] = {
      'features':{},
      'response':{}
    }

  # Generate independent data using the changepoings
  idxSample = 1
  for i in range(len(change_points) + 1):
    try:
      currCp = change_points[i]
    except:
      # There is no next changepoint so we set the num_samples as the uppper limit
      currCp = num_samples + 1 # TODO not sure about this + 1 at the end to fill num_samples
    if i == 0:
      lastCp = idxSample
    else:
      lastCp = change_points[i - 1]
    currSampleLen =  currCp - lastCp

    for j in range(dimensions):
      col_name = 'X' + str(j + 1)
      data['changepoints'][str(i)]['features'][col_name] = np.random.normal(0, 1, currSampleLen)

  # Generate the response Y as a func of the features
  coefs = [
    [2.8, -1.3],
    [1.8, -0.5],
    [3.4, -0,2]
  ]
  # Generate the random noise that will be added to all the samples
  epsilon = np.random.normal(0, 0.1, num_samples)   
  for i in range(len(change_points) + 1):
    # Get the cp length
    try:
      currCp = change_points[i]
    except:
      # There is no next changepoint so we set the num_samples as the uppper limit
      currCp = num_samples + 1 # TODO not sure about this + 1 at the end to fill num_samples
    if i == 0:
      lastCp = idxSample
    else:
      lastCp = change_points[i - 1]
    currSampleLen =  currCp - lastCp

    # Create the response vector y
    currCoefs = coefs[i]
    data['changepoints'][str(i)]['response']['y'] = 0.5 + \
     currCoefs[0] * data['changepoints'][str(i)]['features']['X1'][0:currSampleLen] + \
     currCoefs[1] * data['changepoints'][str(i)]['features']['X2'][0:currSampleLen] + \
     epsilon[0:currSampleLen]
     
  return data

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
  epsilon = np.random.normal(0, 0.1, num_samples) 
  # Create the response vector y
  data['response']['y'] = 0.5 + 2.8 * data['features']['X5'] - 1.3 * data['features']['X2']  + epsilon

  return data

if __name__ == '__main__':
  print('Testing data generation...')
  dat = generateTestDataThird(100, 3)
  print('The dimensions of the response vector are: ')
  #print(dat['response']['y'].shape) old test
  print(len(dat['changepoints']['0']['features']['X1']))
  