import numpy as np

def cpRellocationMove(cpSet):
  '''
    Implements the rellocation move (R) this move will take a changepoint remove it and 
    rellocated inside withing 2 of the adjacent changepoints.

    Args:
      cpSet : list<int>
        list containing the changepoint set with the elements being 
        the positions of the response y.
    Returns:
      cpSetCopy : list<int>
        list with a rellocated changepoint around its neighbors.
    Raises:
      ValueError
        When you try to rellocate a non-existing changepoint.
        Or when the randomly selected cp had 2 consecutive left and right neighbors.
  '''
  
  if len(cpSet) < 2:
    raise ValueError('You need at least two elements on the cpSet to rellocate one') 

  cpSetCopy = cpSet.copy() # Copy in case of mutability
  # TODO improve this
  cpSetCopy[-1] = cpSetCopy[-1] - 1 # Change the last cp to be compatible with new setpoint generation

  indexes = [idx[0] for idx in enumerate(cpSet)] # idx[0] because enumrate() returns a tuple
  indexes.pop(-1) # Remove the last one because that is allways the num_samples
  idxToRellocate = np.random.choice(indexes) # Select one randomly
  
  # Select the adjacent changepoints
  left = idxToRellocate - 1
  if left < 0: # You cannot go past the first cp
    cpLeft = 1 # Min cp needs to be located at 3
  else:
    cpLeft = cpSetCopy[left] # Get the cp to the left

  right = idxToRellocate + 1 # You cannot go past the last cp
  if right == len(cpSet) - 1:
    cpRight = cpSetCopy[-1]  # set at the max - 1 because we do not want the max to be selected twice
  else:
    cpRight = cpSetCopy[right] # Get the cp to the right

  cpSetCopy.pop(idxToRellocate) # Pop the cp that is going to be rellocated
  # Get a list of the possible candidates
  cpCandidates = [idx + cpLeft + 1 for idx in range(cpRight - cpLeft - 1)]
  # In the case of consecutive changepoint you need to filter them just in case
  cpCandidates = list(filter(lambda element: element != cpSet[idxToRellocate], cpCandidates))
  if cpCandidates == []: # Check for consecutive right and left ie, 24, 25, 26 cannot rellocate 25
    raise ValueError('Randomly selected changepoint had 2 consecutive right and left neighbors.')

  #print('the changepoint removed was: ', cpSet[idxToRellocate]) # uncomment for debug
  # Choose one randomly form the candidate list
  newCp = np.random.choice(cpCandidates) # Select one randomly
  cpSetCopy.append(newCp) # Append the new random cp
  cpSetCopy = sorted(cpSetCopy) # Sort the list
  cpSetCopy[-1] = cpSetCopy[-1] + 1 # Add again the 2 so the rest of the algo works TODO improve this
  #print('the changepoint added was: ', newCp) # uncomment for debug
  
  return(cpSetCopy)

def cpDeathMove(cpSet):
  '''
    Implements the Death move (D) of a changepoint, it randomly selects a changepoint from
    the changepoint set and removes it from the data

    Args:
      cpSet : list<int>
        A list containing the changepoint set with the elements being 
        the positions of the response y.

    Returns:
      cpSetCopy : list<int>
        A list that contains the cps without the one that was randomly removed. 
    
    Raises:
      ValueError
        When we try to remove a cpSet that contains < 2 elements.
  '''
  if len(cpSet) < 2:
    raise ValueError('You need at least two elements on the cpSet to remove one') 

  cpSetCopy = cpSet.copy() # Copy in case of mutability
  
  indexes = [idx[0] for idx in enumerate(cpSet)] # idx[0] because enumrate() returns a tuple
  indexes.pop(-1) # Remove the last one because that is allways the num_samples
  idxToRemove = np.random.choice(indexes) # Select one randomly
  cpSetCopy.pop(idxToRemove) # Remove the chosen idx

  return cpSetCopy
  
def cpBirthMove(cpSet, numSamples):
  '''
    Implements the Birth move (B) of a changepoint, it creates a possible set of new
    changepoints given the number of samples and generates one random new one (filtering
    the current ones)

    Args:
      cpSet : list<int> 
        A list containing the changepoint set with the elements being 
        the positions of the response y.
      numSamples : int 
        The total number of samples x1,..x_numSamples
      
    Returns:
      cpSetCopy: list<int>
        A list with the attached randomly generated new changepoint
  '''
  cpSetCopy = cpSet.copy() # Create a copy for possible mutability issues
  # Generate a set of idx for the pc candidates
  candidateCpSet = [idx for idx in range(numSamples + 1)]

  # Remove the first 3 elements [0, 1, 2] because min(cp_{1}) must be at 3
  del candidateCpSet[:2]
  # filter the cps that already exist  
  filteredCandidateCpSet = list(filter(lambda x: x not in cpSet, candidateCpSet))
  newCp = np.random.choice(filteredCandidateCpSet, 1) # Get a random new cp location
  newCp = (newCp).item() # Turn into an scalar

  cpSetCopy.append(newCp) # Append the new random cp
  cpSetCopy = sorted(cpSetCopy) # Sort the list

  return cpSetCopy
