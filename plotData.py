import matplotlib.pyplot as plt
from utils import generateData

def plotScatter(var1, var2, label1, label2):
  # Scatter Plot of the variables 
  plt.scatter(var1, var2, alpha=0.5)
  plt.title('Scatter plot of {0} and {1}'.format(label1, label2))
  plt.xlabel(label1)
  plt.ylabel(label2)
  plt.show()

def plotHistogram(dataVector, label, burnInPeriod = 1000):
  plt.title('Histogram of the Variable {0}'.format(label))
  plt.hist(dataVector[burnInPeriod:], bins='auto')
  plt.show()

def plotTrace(dataVector, label, burnInPeriod = 1000):
  plt.plot(dataVector[burnInPeriod:])
  plt.title('Gibbs sampler trace plot of {0}'.format(label))
  plt.show()

if __name__ == '__main__':
  print('Testing plotting...')
  data = generateData(num_samples = 100, dimensions = 3, dependent = 1)

  plotScatter(data['features']['X1'], data['features']['X2'], 'X1', 'X2' )
