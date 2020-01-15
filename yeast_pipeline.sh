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
  --chain_length 10000
  --burn_in 5000 
  '
# Execute without profiler
python ./examples/yeast_tests.py $ARGS $METHOD