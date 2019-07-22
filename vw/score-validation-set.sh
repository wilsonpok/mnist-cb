#!/bin/bash
set -eux

# Inputs
model=$HOME/mnist-cb/models/model.vw
test_file=$HOME/mnist-cb/data/test.vw

# Outputs
test_preds=$HOME/mnist-cb/data/test_preds.csv



vw \
	--data $test_file \
	--initial_regressor $model \
	--predictions $test_preds \

