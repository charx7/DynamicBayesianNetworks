#! /bin/sh
# Will run the profiler for the current algorithm
# Arguments for profiling on simulated data
export ARGS='
  --num_features 6
  --num_indep 4 
  --num_samples 50 
  --generated_noise_var 1 
  --chain_length 5000 
  --burn_in 1000 
  --lag 1
  --change_points 10 25 
  -v --coefs_file coefs.txt
  --method
  var-glob-dbn
  --lag
  1
'
# Arguments for profiling on yeast data
# export ARGS='
#   --chain_length 1000
#   --burn_in 100
#   --lag 1
#   --method
#   glob-dbn
'

# Execute with profiler
python -m cProfile -o ./output/algorithm_profiling.cprof -s cumulative ./examples/full_parents_test.py $ARGS
# open the profiler
pyprof2calltree -k -i ./output/algorithm_profiling.cprof
