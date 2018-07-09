from scipy import stats

with open('k_fold_AUC_result_logreg.txt', 'r') as f:
    a = [float(line.rstrip('\n')) for line in f]

with open('k_fold_AUC_result_xgb.txt', 'r') as f:
    b = [float(line.rstrip('\n')) for line in f]

with open('k_fold_AUC_result_NN.txt', 'r') as f:
    c = [float(line.rstrip('\n')) for line in f]

print(stats.ttest_rel(a,b))

print(stats.ttest_rel(b,c))