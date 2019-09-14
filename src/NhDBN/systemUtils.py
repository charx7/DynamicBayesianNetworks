import os
import shutil

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
  figures_folder = 'figures/'
  clean_figures_folder(figures_folder)
  #cleanOutput()
  writeStr = 'test String!\n'
  writeOutputFile(writeStr)

if __name__ == '__main__':
  main()
