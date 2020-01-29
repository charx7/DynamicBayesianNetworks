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

def fraction_score(posterior_sample):
  '''
    Compute the fraction score for negative and positive
    sampled coefficients score.
  '''
  post_len = len(posterior_sample)
  # compute the fraction of the samples less than 0
  n_less = len([el for el in posterior_sample if el <= 0])
  n_fraction = n_less / post_len # compute the fraction

  score = 0.5 - n_fraction # substract from 0.5 50% neg coefs and 50% pos coefs
  
  return score

def get_scores_over_time(betas_over_time, currFeatures, dims):
  '''
    Compute the scores over time matrix for the current proposed edges
  '''
  scores_over_time_matrix = np.array([])
  for curr_betas_matrix in betas_over_time:
    time_pt_scores = np.array([]) # empty np array with the current time point scores
    for col_tuple in enumerate(currFeatures):
      jdx = col_tuple[0] + 1
      curr_posterior = curr_betas_matrix[:, jdx]
      #curr_timepoint_score = credible_score(curr_posterior) # for the credible scores bayesian pvalue
      curr_timepoint_score = fraction_score(curr_posterior) # for frac scores
      time_pt_scores = np.append(time_pt_scores, curr_timepoint_score)
    # flatten and append to the overal scores_over_time_matrix
    time_pt_scores = time_pt_scores.reshape(1, dims - 1) # reshape so we can vertically stack
    scores_over_time_matrix = np.concatenate((scores_over_time_matrix, time_pt_scores)) if scores_over_time_matrix.size else time_pt_scores

  return scores_over_time_matrix    

def get_betas_over_time(time_pts, thinned_changepoints, thinned_chain, dims):        
  betas_list = [] # empty list that will contain the 33 thinned chains
  for time_pt in range(time_pts):
    curr_betas_matrix = np.array([]) # declare empty array
    for idx, cps in enumerate(thinned_changepoints):
      concatenated = False
      for jdx, cp in enumerate(cps):
        if (time_pt + 1 < cp and concatenated == False):
          concatenated = True
          time_pt_betas = thinned_chain[idx][jdx].reshape(1, dims) # reshape the curr betas vector
          curr_betas_matrix = np.concatenate((curr_betas_matrix, time_pt_betas)) if curr_betas_matrix.size else time_pt_betas      
    betas_list.append(curr_betas_matrix) # append to the list

  return betas_list

def beta_post_matrix(thinned_chain):
  '''
    Will Construct the beta posteriors matrix
  '''
  # Beta posterior matrix construction
  betas_matrix = np.array([]) # declare an empty np array
  # loop over the chain to create the betas matrix
  for row in thinned_chain:
    # get the beta samples from each segment
    for vec in row:
      r_vec = vec.reshape(1, vec.shape[0]) # reshape for a vertical stack
      betas_matrix = np.concatenate((betas_matrix, r_vec)) if betas_matrix.size else r_vec
      
  return betas_matrix

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

def credible_score(posterior_sample):
  post_len = len(posterior_sample)
  # compute the fraction of the samples larger than 0
  n_greater = len([el for el in posterior_sample if el >= 0])
  # compute the fraction of the samples less than 0
  n_less = len([el for el in posterior_sample if el <= 0])
  # use the maximum of both fractions as our new edge score
  score = max(n_greater / post_len, n_less / post_len)

  return score

def credible_interval(posterior_sample, response, feature, interval_length):
  _ = plt.hist(posterior_sample, bins='auto', density='true')
  title = 'Beta Posterior Sample Histogram for edge ' + str(feature + 1) + \
    ' => ' +  str(response + 1)
  plt.title(title)
  #plt.show() # uncomment for plot show

  # use quantiles to calculate the credible intervals
  lower = interval_length / 2
  upper = (1 - interval_length) / 2
  lower_bound = np.quantile(posterior_sample, lower)
  upper_bound = np.quantile(posterior_sample, upper)

  cred_interval = (lower_bound, upper_bound)
  
  # need to also calculate the p-value
  
  return cred_interval
  
def score_beta_matrix(betas_matrix, currFeatures, currResponse):
  '''
    Calculate the new edge-scores for the
    betas matrix of the full-parents set
  '''
  dims = betas_matrix.shape[1]
  edge_scores = [0 for i in range(dims)] # empty edge scores matrix

  for col_tuple in enumerate(currFeatures):
    idx = col_tuple[0] + 1 # we need to start from 1 because of the intercept
    beta_post = betas_matrix[:, idx] # extract the post sample
    currFeature = col_tuple[1] # get the current feature
    pct = 0.11 # the 1 - percent cred interval
    res = credible_interval(beta_post, currResponse, currFeature, pct) # cred interval computation
    print('The ', 100 - (pct * 100), '% Credible interval for ', currFeature + 1,
      ' -> ', currResponse + 1, ' is: ', res[0], res[1])

    cred_score = credible_score(beta_post) # compute the new score
    edge_scores[currFeature] = cred_score # assign to the score array

    # # We should not check if 0 is in the 95% conf interval (check this)
    # # test of 0 is inside the 95% conf interval -> add 0 to the adj list
    # if not(res[0] <= 0 <= res[1]):
    #   # if we found that 0 is not on the cred interval then we set
    #   # the edge value to 1 
    #   edge_scores[currFeature] = 1
    
  return edge_scores
