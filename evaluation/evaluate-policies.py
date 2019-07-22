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

loss_lookup['loss_a0'] = 1
loss_lookup['loss_a1'] = 1
loss_lookup['loss_a2'] = 1
loss_lookup['loss_a3'] = 1
loss_lookup['loss_a4'] = 1
loss_lookup['loss_a5'] = 1
loss_lookup['loss_a6'] = 1
loss_lookup['loss_a7'] = 1
loss_lookup['loss_a8'] = 1
loss_lookup['loss_a9'] = 1

loss_lookup['loss_a0'].loc[loss_lookup.actual == 0] = -1
loss_lookup['loss_a1'].loc[loss_lookup.actual == 1] = -1
loss_lookup['loss_a2'].loc[loss_lookup.actual == 2] = -1
loss_lookup['loss_a3'].loc[loss_lookup.actual == 3] = -1
loss_lookup['loss_a4'].loc[loss_lookup.actual == 4] = -1
loss_lookup['loss_a5'].loc[loss_lookup.actual == 5] = -1
loss_lookup['loss_a6'].loc[loss_lookup.actual == 6] = -1
loss_lookup['loss_a7'].loc[loss_lookup.actual == 7] = -1
loss_lookup['loss_a8'].loc[loss_lookup.actual == 8] = -1
loss_lookup['loss_a9'].loc[loss_lookup.actual == 9] = -1




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



comparison['random'] = comparison['pred'].sample(frac=1, random_state=666)\
.reset_index(drop=True)


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
# IPS
###########
round(np.mean(model_set['loss_w']), 5)
# -9.98366

round(np.mean(random_set['loss_w']), 5)
# -8.05594
