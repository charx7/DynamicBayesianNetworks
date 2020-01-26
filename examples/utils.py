import os
import shutil
import pathlib
import pickle
import logging
import matplotlib.pyplot as plt

from pprint import pprint
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from numpy import genfromtxt

# Logger configuration TODO move this into a config file
logger = logging.getLogger(__name__) # create a logger obj
logger.setLevel(logging.INFO) # establish logging level
# Establish the display of the logger
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s') 
file_handler = logging.FileHandler('output.log', mode='a') # The file output name
# Add the formatter to the logger
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def transformResults(trueAdjMatrix, adjMatrixProp):
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
  
  trueAdjMatrix = flattened_true
  adjMatrixProp = flattened_scores

  return trueAdjMatrix, adjMatrixProp

def scoreMetrics(adjMatrixProp, trueAdjMatrix):
  # calculate various metrics
  precision = precision_score(trueAdjMatrix, adjMatrixProp)
  recall = recall_score(trueAdjMatrix, adjMatrixProp)
  f1 = f1_score(trueAdjMatrix, adjMatrixProp)

  print('The precision for the 95 percent classifier is:', precision)
  print('The recall for the 95 percent is:', recall)
  print('The f1-score for the 95 percent classifier is:', f1)

def adjMatrixRoc(adjMatrixProp, trueAdjMatrix, verbose):
  if verbose:
    print('The true adj matrix is: ') ; logger.info('The true adj matrix is: ')
    pprint(trueAdjMatrix) ; logger.info(str(trueAdjMatrix))
    print('The proposed adj matrix is: ') ; logger.info('The proposed adj matrix is: ')
    pprint(adjMatrixProp) ; logger.info(str(adjMatrixProp))
  
  drawRoc(adjMatrixProp, trueAdjMatrix) # Draw the RoC curve
  drawPRC(adjMatrixProp, trueAdjMatrix) # Draw the PR curve

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
    
def data_reader(data_dir):
  # get the current path
  path = pathlib.Path.cwd()
  # Path handling with the debugger
  clPath = path.joinpath('examples')
    
  # Try catch block to either run on the console or on the debugger/root folder
  try: # try that we are executing on the same path as the file is
    np_data = genfromtxt(path.joinpath(data_dir), delimiter=',')
  except: # we are executing from the root of our project 
    np_data = genfromtxt(clPath.joinpath(data_dir), delimiter=',')
  return np_data  # return the data

def clean_figures_folder(figures_folder):
  '''
      Will remove any previous results from the working directory and build the
      figures directory again

      Args:
          figures_folder : str
              directory name 
      Returns:
          void
  '''
  try:
    # Remove the pre-existing figures folder
    shutil.rmtree(figures_folder)
  except:
    print("No figures folder")
  # Create again the figures folder
  os.mkdir(figures_folder)


def cleanOutput():
  '''
    Cleans the output folder from previous runs.
  '''
  # Clean the output folder
  try:
    # Remove the pre-existing output folder
    shutil.rmtree('./output')
  except:
    print("No output folder")
  
  # Make the output directory again
  os.mkdir('output/')  

def save_chain(filename, network_object):
  '''
    Saves the network object into the output folder.
    
    Args:
      filename : str
        the name of the file
      network_object : Network
        object that is going to be saved into disc
  '''
  filepath = os.path.join('./output/',filename) # append to the filepath

  # save with pickle
  with open(filepath, 'wb') as f:
    pickle.dump(network_object, f)

def load_chain(filename):
  '''
    Loads the chain file that has been previouly saved in the
    /output/ folder.

    Args:
      filename : str
        name of the file to read
  '''
  filepath = os.path.join('./output/',filename) # append to the filepath

  try:
    pckled_obj = open(filepath, 'rb')
    network = pickle.load(pckled_obj)
    pckled_obj.close()
    print('Obj succesfully loaded.')
    return network
  except:
    print('No file was loaded, please try again.')
    return

def writeOutputFile(text = ''):
  '''
    Saves the output to a text file containing the results.
      
      Args:
        text : str
          string of text to append to the output file

      Returns:
        void : void
  '''
  if os.path.isfile('./output/output.txt'): # check if output file already exists
    with open(os.path.join('output/output.txt'), "a") as f: # Open the output
      f.write(text) # append text str
  else:
    # Create a new output file on the output directory
    with open(os.path.join('output/output.txt'), 'w') as output:
        output.write('Output file: \n')
        
def main():
  data = data_reader('./data/datayeastoff.txt')
  print(data)

if __name__ == '__main__':
  main()
