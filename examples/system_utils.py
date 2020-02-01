import os
import subprocess

def clean_output():
  print('Cleaning the output folder...')
  cmd = ['sh', 'clean_latex.sh']
  proc = subprocess.Popen(cmd) # call a subprocess and pass it the args
  
def main():
  # arguments that will format the .tex
  args = {
    'title':'testerino'
  }

  # string that defines the body of the latex document
  content = r'''\documentclass[a4paper]{article}
  \usepackage[a4paper,top=2cm,bottom=2.5cm,left=1.5cm,right=1.5cm,marginparwidth=1.75cm]{geometry}
  \usepackage[english]{babel} 
  \usepackage[utf8x]{inputenc}

  \usepackage{float}
  \usepackage{amsmath}
  \usepackage{graphicx}
  \graphicspath{ {./images/} }
  \title{''' + args['title'] + r'''}

  \begin{document}
  \maketitle

  \section*{Model Output}
  ------------ \\
  Model: Bayesian Non-Homogeneous \\
  type: full parents \\
  length: 30k \\
  burn-in: 10k \\
  thinning: modulo 10 \\
  ------------ \\
  AUPRC: 0.56 \\
  scoring method: frac\_score \\
  \begin{figure}[ht]
    \includegraphics[width=10cm]{./figures/prc.png}
    \centering
    \caption{Area Under the Precision Recall Curve} 
  \end{figure}
  \subsection{Diagnostics}
  Network Configuration: * = x1 \\
  Residuals over time Boxplot

  \end{document}'''

  # write the codument into disc
  with open('./output/model_diagnostics.tex','w') as f:
    f.write(content%args)

  # latex compilation
  cmd = ['pdflatex','-output-directory=./output', '-interaction', 'nonstopmode', './output/model_diagnostics.tex']
  proc = subprocess.Popen(cmd) # call a subprocess and pass it the args
  proc.communicate()

  retcode = proc.returncode # return code to see if the compilation was succesful
  if not retcode == 0:
    os.unlink('./output/model_diagnostics.pdf') # unlink just in case
    raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd))) 
  
  # unlink the files so we dont eat up memory
  os.unlink('./output/model_diagnostics.tex')
  
  # clean the output
  clean_output()

if __name__ == '__main__':
  main()
  