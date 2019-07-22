import pandas as pd
import numpy as np
from pathlib import Path

# Inputs
test_preds_file = str(Path.home()) + '/mnist-cb/data/test_preds.csv'
test_file = str(Path.home()) + '/mnist-cb/data/test.csv'


#####################
# Load data
#####################

# Test data
test = pd.read_csv(test_file)

# Test predictions
test_preds = pd.read_csv(test_preds_file, header=None)
test_preds.columns = ['pred_vw']
test_preds['model'] = test_preds['pred_vw'] - 1

[x.shape for x in [test, test_preds]]
# [(10000, 785), (10000, 1)]




test_preds['model'].value_counts(normalize=True).sort_index()
# 0    0.1036
# 1    0.1161
# 2    0.1003
# 3    0.1016
# 4    0.1006
# 5    0.0849
# 6    0.0972
# 7    0.1023
# 8    0.0942
# 9    0.0992

test['label'].value_counts(normalize=True).sort_index()
# _0    0.0980
# _1    0.1135
# _2    0.1032
# _3    0.1010
# _4    0.0982
# _5    0.0892
# _6    0.0958
# _7    0.1028
# _8    0.0974
# _9    0.1009



#####################
# Loss lookup
#####################

loss_lookup = pd.DataFrame({'actual' : test['label'].str.replace('_', '').astype(int)})

loss_for_mismatch = 0
c = -1

loss_lookup['loss_a0'] = loss_for_mismatch
loss_lookup['loss_a1'] = loss_for_mismatch
loss_lookup['loss_a2'] = loss_for_mismatch
loss_lookup['loss_a3'] = loss_for_mismatch
loss_lookup['loss_a4'] = loss_for_mismatch
loss_lookup['loss_a5'] = loss_for_mismatch
loss_lookup['loss_a6'] = loss_for_mismatch
loss_lookup['loss_a7'] = loss_for_mismatch
loss_lookup['loss_a8'] = loss_for_mismatch
loss_lookup['loss_a9'] = loss_for_mismatch

loss_lookup['loss_a0'].loc[loss_lookup.actual == 0] = c
loss_lookup['loss_a1'].loc[loss_lookup.actual == 1] = c
loss_lookup['loss_a2'].loc[loss_lookup.actual == 2] = c
loss_lookup['loss_a3'].loc[loss_lookup.actual == 3] = c
loss_lookup['loss_a4'].loc[loss_lookup.actual == 4] = c
loss_lookup['loss_a5'].loc[loss_lookup.actual == 5] = c
loss_lookup['loss_a6'].loc[loss_lookup.actual == 6] = c
loss_lookup['loss_a7'].loc[loss_lookup.actual == 7] = c
loss_lookup['loss_a8'].loc[loss_lookup.actual == 8] = c
loss_lookup['loss_a9'].loc[loss_lookup.actual == 9] = c




#####################
# Compare policies
#####################

comparison = pd.concat([loss_lookup, test_preds['model']], axis=1)

comparison['random'] = comparison['model'].sample(frac=1, random_state=666)\
.reset_index(drop=True)


len(comparison.loc[comparison.actual == comparison.model]) / len(comparison)
# 0.9059

len(comparison.loc[comparison.actual == comparison.random]) / len(comparison)
# 0.0989



# Model loss
comparison['loss_model'] = np.nan
comparison['loss_model'].loc[comparison.model == 0] = comparison.loss_a0
comparison['loss_model'].loc[comparison.model == 1] = comparison.loss_a1
comparison['loss_model'].loc[comparison.model == 2] = comparison.loss_a2
comparison['loss_model'].loc[comparison.model == 3] = comparison.loss_a3
comparison['loss_model'].loc[comparison.model == 4] = comparison.loss_a4
comparison['loss_model'].loc[comparison.model == 5] = comparison.loss_a5
comparison['loss_model'].loc[comparison.model == 6] = comparison.loss_a6
comparison['loss_model'].loc[comparison.model == 7] = comparison.loss_a7
comparison['loss_model'].loc[comparison.model == 8] = comparison.loss_a8
comparison['loss_model'].loc[comparison.model == 9] = comparison.loss_a9


# Random loss
comparison['loss_random'] = np.nan
comparison['loss_random'].loc[comparison.random == 0] = comparison.loss_a0
comparison['loss_random'].loc[comparison.random == 1] = comparison.loss_a1
comparison['loss_random'].loc[comparison.random == 2] = comparison.loss_a2
comparison['loss_random'].loc[comparison.random == 3] = comparison.loss_a3
comparison['loss_random'].loc[comparison.random == 4] = comparison.loss_a4
comparison['loss_random'].loc[comparison.random == 5] = comparison.loss_a5
comparison['loss_random'].loc[comparison.random == 6] = comparison.loss_a6
comparison['loss_random'].loc[comparison.random == 7] = comparison.loss_a7
comparison['loss_random'].loc[comparison.random == 8] = comparison.loss_a8
comparison['loss_random'].loc[comparison.random == 9] = comparison.loss_a9




# Action probabilities
probs = test['label'].value_counts(normalize=True).sort_index().reset_index(drop=True)
probs = probs.reset_index()
probs.columns = ['actual', 'prob']

comparison_prob = comparison.merge(probs, how='outer')



# Filter sets
model_set = comparison_prob.loc[comparison_prob.actual == comparison_prob.model]
random_set = comparison_prob.loc[comparison_prob.actual == comparison_prob.random]

len(model_set)
# 9059

len(random_set)
# 989


###########
# DM
###########
dm_model = np.sum(comparison_prob['loss_model']) / len(comparison_prob)
dm_model
# -0.9059

dm_random = np.sum(comparison_prob['loss_random']) / len(comparison_prob)
dm_random
# -0.0989


###########
# IPS
###########
np.sum(model_set['loss_model'] / model_set['prob']) / len(comparison_prob)
# -9.044200016780616

np.sum(random_set['loss_random'] / model_set['prob']) / len(comparison_prob)
# -0.8909614311352875


###########
# DR
###########
model_set['dr_calc'] = ((model_set['loss_model'] - dm_model) / model_set['prob']) + dm_model
np.sum(model_set['dr_calc']) / len(comparison_prob)
# -1.671714031579056

random_set['dr_calc'] = ((random_set['loss_random'] - dm_random) / random_set['prob']) + dm_random
np.sum(random_set['dr_calc']) / len(comparison_prob)
# -0.8980965263979739
