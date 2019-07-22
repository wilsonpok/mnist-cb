#!/bin/bash
set -eux

# Inputs
train_file=$HOME/mnist-cb/data/train.vw

# Outputs
train_preds=$HOME/mnist-cb/data/train_preds.csv
model=$HOME/mnist-cb/models/model.vw
readable_model=$HOME/mnist-cb/models/model-readable.txt


vw \
	--data $train_file \
	--oaa 10 \
	--predictions $train_preds \
	--final_regressor $model \
	--readable_model $readable_model
