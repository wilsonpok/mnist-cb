import pandas as pd
import numpy as np
from pathlib import Path

# Inputs
test_preds_file = str(Path.home()) + '/mnist-cb/data/test_preds.csv'
test_file = str(Path.home()) + '/mnist-cb/data/test.csv'



#####################
# Load data
#####################

test = pd.read_csv(test_file)


test_preds = pd.read_csv(test_preds_file, header=None)
test_preds.columns = ['pred_vw']
test_preds['pred'] = test_preds['pred_vw'] - 1





[x.shape for x in [test, test_preds]]
# [(10000, 785), (10000, 1)]

test_preds['pred'].value_counts(normalize=True)
# 1    0.1161
# 0    0.1036
# 7    0.1023
# 3    0.1016
# 4    0.1006
# 2    0.1003
# 9    0.0992
# 6    0.0972
# 8    0.0942
# 5    0.0849

test['label'].value_counts(normalize=True)
# _1    0.1135
# _2    0.1032
# _7    0.1028
# _3    0.1010
# _9    0.1009
# _4    0.0982
# _0    0.0980
# _8    0.0974
# _6    0.0958
# _5    0.0892



#####################
# Loss lookup
#####################

loss_lookup = pd.DataFrame({'actual' : test['label'].str.replace('_', '').astype(int)})

loss_for_mismatch = 0

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

c = -1

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

comparison = pd.concat([loss_lookup, test_preds['pred']], axis=1)

len(comparison.loc[comparison.actual == comparison.pred]) / len(comparison)
# 0.9059


comparison['loss'] = 0
comparison['loss'].loc[comparison.pred == 0] = comparison.loss_a0
comparison['loss'].loc[comparison.pred == 1] = comparison.loss_a1
comparison['loss'].loc[comparison.pred == 2] = comparison.loss_a2
comparison['loss'].loc[comparison.pred == 3] = comparison.loss_a3
comparison['loss'].loc[comparison.pred == 4] = comparison.loss_a4
comparison['loss'].loc[comparison.pred == 5] = comparison.loss_a5
comparison['loss'].loc[comparison.pred == 6] = comparison.loss_a6
comparison['loss'].loc[comparison.pred == 7] = comparison.loss_a7
comparison['loss'].loc[comparison.pred == 8] = comparison.loss_a8
comparison['loss'].loc[comparison.pred == 9] = comparison.loss_a9







comparison['random'] = comparison['pred'].sample(frac=1, random_state=666)\
.reset_index(drop=True)

comparison['loss_random'] = 0
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





p0 = 0.0980
p1 = 0.1135
p2 = 0.1032
p3 = 0.1010
p4 = 0.0982
p5 = 0.0892
p6 = 0.0958
p7 = 0.1028
p8 = 0.0974
p9 = 0.1009


comparison['prob'] = 0
comparison['prob'].loc[comparison.pred == 0] = p0
comparison['prob'].loc[comparison.pred == 1] = p1
comparison['prob'].loc[comparison.pred == 2] = p2
comparison['prob'].loc[comparison.pred == 3] = p3
comparison['prob'].loc[comparison.pred == 4] = p4
comparison['prob'].loc[comparison.pred == 5] = p5
comparison['prob'].loc[comparison.pred == 6] = p6
comparison['prob'].loc[comparison.pred == 7] = p7
comparison['prob'].loc[comparison.pred == 8] = p8
comparison['prob'].loc[comparison.pred == 9] = p9


comparison['prob_random'] = 0
comparison['prob_random'].loc[comparison.random == 0] = p0
comparison['prob_random'].loc[comparison.random == 1] = p1
comparison['prob_random'].loc[comparison.random == 2] = p2
comparison['prob_random'].loc[comparison.random == 3] = p3
comparison['prob_random'].loc[comparison.random == 4] = p4
comparison['prob_random'].loc[comparison.random == 5] = p5
comparison['prob_random'].loc[comparison.random == 6] = p6
comparison['prob_random'].loc[comparison.random == 7] = p7
comparison['prob_random'].loc[comparison.random == 8] = p8
comparison['prob_random'].loc[comparison.random == 9] = p9



comparison['loss_w'] = comparison.loss / comparison.prob
comparison['loss_w_random'] = comparison.loss_random / comparison.prob_random

model_set = comparison.loc[comparison.actual == comparison.pred]
random_set = comparison.loc[comparison.actual == comparison.random]


len(model_set)
# 9059

len(random_set)
# 989


###########
# DM
###########
dm_model = np.sum(comparison['loss']) / len(comparison)
dm_model
# -0.9059

dm_random = np.sum(comparison['loss_random']) / len(comparison)
dm_random
# -0.0989


###########
# IPS
###########
np.sum(comparison['loss_w'].loc[comparison.actual == comparison.pred]) / len(comparison)
# -9.044200016780616

np.sum(comparison['loss_w_random'].loc[comparison.actual == comparison.random]) / len(comparison)
# -0.9858121367195357


###########
# DR
###########

model_set = comparison.loc[comparison.actual == comparison.pred]
model_set['dr_calc'] = ((model_set['loss'] - dm_model) / model_set['prob']) + dm_model
np.sum(model_set['dr_calc']) / len(comparison)
# -1.671714031579056

random_set = comparison.loc[comparison.actual == comparison.random]
random_set['dr_calc'] = ((random_set['loss'] - dm_random) / random_set['prob']) + dm_random
np.sum(random_set['dr_calc']) / len(comparison)
# -0.80330733049238




