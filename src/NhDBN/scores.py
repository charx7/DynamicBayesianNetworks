import numpy as np

def calculateFeatureScores(selectedFeaturesVector, totalDims):
  for i in range(totalDims):
    print('Calculating score for {0}'.format(i+1))
    currentFeature = i + 1 

    freqSum = 0
    # Calculate the % of apperance
    for currentPi in selectedFeaturesVector:
      if currentFeature in currentPi:
        freqSum = freqSum + 1
    
    print(freqSum / len(selectedFeaturesVector))

def testFeatureScores():
  print('Executing test for calculation features scores...')
  dummyData = [
    np.array([1, 3, 6]),
    np.array([1, 3]),
    np.array([8]),
    np.array([9 , 10])
  ]
  
  calculateFeatureScores(dummyData, 10)

if __name__ == '__main__':
  testFeatureScores()
  