import os
import subprocess

def clean_output():
  print('Cleaning the output folder...')
  cmd = ['sh', 'clean_latex.sh']
  proc = subprocess.Popen(cmd) # call a subprocess and pass it the args
  
def generate_report(args, title, auprc):
  '''
    Will generate the report on a pdf format with latex
  '''
  args['title'] = title # add the title arg
  args['auprc'] = '0.80' # add the auprc arg
  
  # arguments that will format the .tex (dummy data for now)
  # args = {
  #   'title':'Model Output',
  #   'model': 'Bayesian Non-Homogeneous',
  #   'type': 'Full-Parents',
  #   'length': '30k',
  #   'burn-in': '10k',
  #   'thinning': 'modulo 10',
  #   'auprc': '0.80',
  #   'scoring_method': 'frac-score',
  #   'network_configs': [
  #     {
  #       'features': ['X1', 'X2','X3','X4'],
  #       'response': 'X0' 
  #     },
  #     {
  #       'features': ['X0', 'X2', 'X3', 'X4'],
  #       'response': 'X1'
  #     }
  #   ]
  # }

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
  Model: ''' + args['model'] + r''' \\
  type: ''' + args['type'] + r''' \\
  length: ''' + args['length'] + r''' \\
  burn-in: ''' + args['burn-in'] + r''' \\
  thinning: ''' + args['thinning'] + r''' \\
  ------------ \\
  AUPRC: ''' + args['auprc'] + r''' \\
  scoring method: ''' + args['scoring_method'] + r'''  \\
  \begin{figure}[ht]
    \includegraphics[width=10cm]{./figures/prc.png}
    \centering
    \caption{Area Under the Precision Recall Curve} 
  \end{figure}
  \section{Diagnostics}'''

  # loop for every configuration 
  for network_config in args['network_configs']:
    curr_feats = network_config['features']
    curr_resp = network_config['response']
    curr_content = r'''
    \subsection{Network Configuration: y =''' + curr_resp + r'''}
    \subsubsection{Residuals over time Boxplot}
    \begin{figure}[H]
      \includegraphics[width=10cm]{./figures/residuals_over_time_''' + curr_resp + r'''.png}
      \centering
      \caption{Residuals over time} 
    \end{figure}
    \subsubsection{Change-Points Probability Barplot}
    \begin{figure}[H]
      \includegraphics[width=10cm]{./figures/changepoints_prob_''' + curr_resp + r'''.png}
      \centering
      \caption{Probability of a changepoint.} 
    \end{figure}
    '''
    content = content + curr_content # append to the latex file
    if type == 'Full-Parents':
      # we have this here because the plots may not be applicable on other models
      content = content + r'''\subsubsection{Edge Specific Plots}''' 
      # Now we have to loop over the edge specific plots
      for edge in curr_feats:
        curr_content = r'''
        \textbf{Edge: ''' + edge + r'''--''' + curr_resp +r'''}
        \\Edge betas over time plot
        \begin{figure}[H]
          \includegraphics[width=10cm]{./figures/edge_''' + edge.lstrip('X') + r'''_''' + curr_resp.lstrip('X') + r'''_boxplot_betas_overtime.png}
          \centering
          \caption{Betas over time bloxplot.} 
        \end{figure}
        Edge fraction score over time plot
        \begin{figure}[H]
          \includegraphics[width=10cm]{./figures/edge_fraction_''' + edge.lstrip('X') + r'''_''' + curr_resp.lstrip('X') + r'''.png}
          \centering
          \caption{Edge Fraction-Score overtime plot.} 
        \end{figure}
        '''
        content = content + curr_content # append to the .tex file

  footer = r'''\end{document}''' # footer of the doc
  content = content + footer

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
  generate_report()
  