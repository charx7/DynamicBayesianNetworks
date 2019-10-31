#! /bin/sh
# Will run the algorithm
export ARGS='
  --num_features 6
  --num_indep 4 
  --num_samples 50 
  --generated_noise_var 1.2
  --chain_length 5000 
  --burn_in 1000 
  --change_points 10 25 
  -v --coefs_file coefs.txt
  '
# Execute without profiler
python ./src/NhDBN/main.py $ARGS
