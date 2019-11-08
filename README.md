# dyban
This package is intended to be used for Network Reconstruction of Dynamic Bayesian Networks.

To test the algorithm on the Yeast data set run the bash script.
Example to run a Non-Homogeneous Dynamic Bayesian Network
```
  sh yeast_pipeline.sh -m nh-dbn
```
Where -m denotes the method to use
- 'h-dbn' -> Homogeneous Dynamic Bayesian Network
- 'nh-dbn' -> Non-Homonegeneous Dynamic Bayesian Network
- 'seq-dbn' -> Sequentially Coupled Dynamic Bayesian Network
- 'glob-dbn' -> Globally Coupled Dynamic Bayesian Network

This will be the readme of the package. 
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
for reference.

- Run this command to install the package locally

```
  pip install .
```

Or 

```
  pip install -e .
```
To be able to edit the source code and (hot-reload) updates?

To run the python profiler use the bash script:
```
  sh algorithm_profiling.sh
```
In order to be able to see the profiler results you need to have 'kcachegrind'
```
   sudo apt-get install -y kcachegrind 
```
