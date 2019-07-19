# https://pjreddie.com/projects/mnist-in-csv/

import pandas as pd

################
# Load data
################

train = pd.read_csv('https://pjreddie.com/media/files/mnist_train.csv',
  header=None)

test = pd.read_csv('https://pjreddie.com/media/files/mnist_test.csv',
  header=None)

[x.shape for x in [train, test]]
# [(60000, 785), (10000, 785)]


################
# Output data
################

train.to_csv('~/mnist-cb/data/train.csv', index=False)
test.to_csv('~/mnist-cb/data/test.csv', index=False)
