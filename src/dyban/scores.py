import numpy as np
import logging
import matplotlib.pyplot as plt
from .systemUtils import clean_figures_folder, writeOutputFile
from pprint import pprint
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Logger configuration TODO move this into a config file
logger = logging.getLogger(__name__) # create a logger obj
logger.setLevel(logging.INFO) # establish logging level
# Establish the display of the logger
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s') 
file_handler = logging.FileHandler('output.log', mode='a') # The file output name
# Add the formatter to the logger
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def adjMatrixRoc(adjMatrixProp, trueAdjMatrix, verbose):
  if verbose:
    print('The true adj matrix is: ') ; logger.info('The true adj matrix is: ')
    pprint(trueAdjMatrix) ; logger.info(str(trueAdjMatrix))
    print('The proposed adj matrix is: ') ; logger.info('The proposed adj matrix is: ')
    pprint(adjMatrixProp) ; logger.info(str(adjMatrixProp))
  # Remove the diagonal that is allways going to be right
  trueAdjMatrixNoDiag = []
  idxToRemove = 0
  for row in trueAdjMatrix:
    row.pop(idxToRemove)
    trueAdjMatrixNoDiag.append(row)
    idxToRemove = idxToRemove + 1
  # Now for the inferred matrix  
  adjMatrixPropNoDiag = []
  idxToRemove = 0
  for row in adjMatrixProp:
    row.pop(idxToRemove)
    adjMatrixPropNoDiag.append(row)
    idxToRemove = idxToRemove + 1
  # Re-assign them
  trueAdjMatrix = trueAdjMatrixNoDiag
  adjMatrixProp = adjMatrixPropNoDiag

  # Flatten the adj matrix to pass to the RoC
  flattened_true = [item for sublist in trueAdjMatrix for item in sublist]
  flattened_true = [1 if item else 0 for item in flattened_true] # convert to binary response vector
  flattened_scores = [item for sublist in adjMatrixProp for item in sublist]
  
  drawRoc(flattened_scores, flattened_true) # Draw the RoC curve
  drawPRC(flattened_scores, flattened_true) # Draw the PR curve

def drawPRC(inferredScoreEdges, realEdges):
  precision, recall, _ = precision_recall_curve(realEdges, inferredScoreEdges)

  # calculate precision/recall auc
  auc_prec_recall = auc(recall, precision)

  print('The AuC of the PR curve was: ', auc_prec_recall)
  plt.clf() # clear previous figure
  plt.title('Precision-Recall Curve')
  plt.plot(recall, precision, marker='.', label='AUC = %0.2f' % auc_prec_recall)
  
  # axis labels
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  # axis limits
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  # show the legend
  plt.legend()
  # save the plot
  figure_route = 'figures/prc'
  plt.savefig(figure_route, bbox_inches='tight')
  plt.show()

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
  #plt.show() #uncomment to show the figure on finish
  clean_figures_folder('figures/')
  figure_route = 'figures/roc'
  plt.savefig(figure_route, bbox_inches='tight')

def calculateFeatureScores(selectedFeaturesVector, totalDims, currentFeatures, currentResponse):
  adjRow = [0 for x in range(totalDims)]
  
  # Print and write the output
  output_line = (
    '>> The current response feature is: X{0}'.format(currentResponse + 1)
  )
  print(output_line) ; logger.info(output_line)

  results = {}
  for feat in currentFeatures:
    output_line = (
      'Edge score for X{0}: '.format(feat + 1)
    )
    print(output_line) ; logger.info(output_line)
    freqSum = 0
    # Calculate the % of apperance
    for currentPi in selectedFeaturesVector:
      if feat in currentPi:
        freqSum = freqSum + 1
    
    denom = len([x for x in selectedFeaturesVector if x.size != 0])
    # Append to the dictionary of the results
    results['X' + str(feat + 1)] = freqSum / denom
    output_line = (
      str(results['X' + str(feat + 1)]) + '\n'
    )
    print(output_line) ; logger.info(output_line)
    # Better return a row on the proposed adj matrix
    adjRow[feat] = freqSum / denom

  return adjRow
    