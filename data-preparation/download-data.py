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

train['label'] = '_' + train[0].astype(str)
test['label'] = '_' + test[0].astype(str)


################
# Output data
################

train_output = pd.concat([train['label'], train.drop([0, 'label'], axis=1)],
  axis=1)

test_output = pd.concat([test['label'], test.drop([0, 'label'], axis=1)],
  axis=1)


train_output.to_csv('~/mnist-cb/data/train.csv', index=False)
test_output.to_csv('~/mnist-cb/data/test.csv', index=False)
