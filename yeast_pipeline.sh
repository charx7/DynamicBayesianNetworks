#! /bin/sh
# Argument parsing for the models
while getopts m: option
  do
    case "${option}" in
    m) METHOD=${OPTARG};;
  esac
done

METHOD="-m ${METHOD}"
echo 'Running the: '$METHOD

export ARGS=' 
  --chain_length 30000
  --burn_in 15000 
  --change_points 15
  '
# Execute without profiler
python ./examples/yeast_tests.py $ARGS $METHOD