#! /bin/sh
# Will run the algorithm
export ARGS=' 
  --chain_length 30000 
  --burn_in 20000 
  --change_points 15
  '
# Execute without profiler
python ./src/NhDBN/yeast_tests.py $ARGS
