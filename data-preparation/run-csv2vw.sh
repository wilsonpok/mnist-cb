#!/bin/bash
set -eux

# Inputs
train_file=$HOME/mnist-cb/data/train.csv
test_file=$HOME/mnist-cb/data/test.csv

# Outputs
train_vw_file=$HOME/mnist-cb/data/train.vw
test_vw_file=$HOME/mnist-cb/data/test.vw


cd $HOME/mnist-cb/data-preparation


python csv2vw.py \
	--skip_headers \
	--convert_zeros \
	$train_file \
	$train_vw_file


python csv2vw.py \
	--skip_headers \
	--convert_zeros \
	$test_file \
	$test_vw_file
