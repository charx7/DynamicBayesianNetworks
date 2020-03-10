import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils import load_chain, transformResults

def main():
  true_inc = [
    [0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0 ,0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0]
  ]

  nmbr_chains = 5
  model_name = 'yeast_vv_glob_coup_dbn'
  chains = []
  for idx in range(nmbr_chains):
    file_name = model_name + '_' + str(idx + 1) + '.pckl'
    curr_chain = load_chain(file_name)
    
    _, flattened_scores = transformResults(true_inc, curr_chain.proposed_adj_matrix)
    chains.append(flattened_scores)

  colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:blue']
  labels = ['chain_1', 'chain_2', 'chain_3', 'chain_4', 'chain_5']
  fig, ax = plt.subplots()
  for idx, edge_scores in enumerate(chains):
    _x = [x + 1 for x in range(len(edge_scores))]
    color = colors[idx] 
    label = labels[idx]
    ax.scatter(_x, edge_scores, c=color, label=label,
               alpha=0.3, edgecolors='none')
    ax.xaxis.set_major_locator(MaxNLocator(integer = True)) # force x ticks as integers

  ax.legend()
  ax.grid(True)

  plt.xlabel('Edge Number')
  plt.ylabel('Edge Score')
  plt.show()
  

if __name__ == '__main__':
  main()
