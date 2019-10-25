import os
import shutil
import pathlib
from numpy import genfromtxt

def data_reader(data_dir):
  # get the current path
  path = pathlib.Path.cwd()
  # Path handling with the debugger
  clPath = path.joinpath('src', 'NhDBN')
    
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
