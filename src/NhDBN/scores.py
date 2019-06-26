import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def drawRoc(inferredScoreEdges, realEdges):
  # Calculate false positive rate and true positive rate
  fpr, tpr, threshold = roc_curve(realEdges, inferredScoreEdges)
  roc_auc = auc(fpr, tpr)
  # Plot the RoC curve
  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr, marker = 'D', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  #plt.xlim([0, 1])
  #plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

def calculateFeatureScores(selectedFeaturesVector, totalDims, currentFeatures, currentResponse):
  adjRow = [0 for x in range(totalDims)]
  print('The current response feature is: X{0}'.format(currentResponse + 1))
  results = {}
  for feat in currentFeatures:
    print('Calculating score for X{0}'.format(feat + 1)) 
    freqSum = 0
    # Calculate the % of apperance
    for currentPi in selectedFeaturesVector:
      if feat in currentPi:
        freqSum = freqSum + 1
    
    # Append to the dictionary of the results
    results['X' + str(feat + 1)] = freqSum / len(selectedFeaturesVector)
    print(results['X' + str(feat + 1)])
    # Better return a row on the proposed adj matrix
    adjRow[feat] = freqSum / len(selectedFeaturesVector)

  return adjRow
    
def testFeatureScores():
  dummyData = [
    np.array([1, 3, 6]),
    np.array([1, 3]),
    np.array([8]),
    np.array([9 , 10])
  ]
  
  calculateFeatureScores(dummyData, 10)

def testRocDraw():
  dummyData = {
    'X1': 0.11,
    'X2': 0.99,
    'X3': 0.10,
    'X4': 0.50,
    'X5': 0.99,
    'X6': 0.10
  }

  realEdges = {
    'X1': 0,
    'X2': 1,
    'X3': 0,
    'X4': 0,
    'X5': 1,
    'X6': 0
  }

  y_score = np.array(list(dummyData.values()))
  y_real = np.array(list(realEdges.values()))
  # Test func
  drawRoc(y_score, y_real)

if __name__ == '__main__':
  #testFeatureScores()
  testRocDraw()
  