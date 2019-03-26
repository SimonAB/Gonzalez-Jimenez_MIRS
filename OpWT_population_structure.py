#%% import modules
import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
import ast

#%% load data
df_test = pd.read_table("test.csv");
df_test_intervention = pd.read_table("test_intervention.csv");
age_prop = pd.read_csv("prop.table.csv")

# load trained models
rskf_results = pd.read_csv("./results/rskf_results.csv")

# use best classifier for predictions
best_clf = rskf_results.sort_values(by="Accuracy", ascending=False).iloc[0, 3]
best_clf = pickle.loads(ast.literal_eval(best_clf))

age_prop_pred = best_clf.predict(df_test)
age_prop_int_pred = best_clf.predict(df_test_intervention)

# Predict and reconstruct age structure of population post intervention

# Test distribution of count data and predicted
true = df_test_intervention.index.values
pred = age_prop_int_pred
ks_fit_int = stats.ks_2samp(true, pred)
stats.chisquare(f_obs=pred, f_exp=true)

# Test of half-ogistic fits
hl_true = stats.halflogistic.fit(true)
hl_pred = stats.halflogistic.fit(pred)
hl_fit_int = stats.ks_2samp(hl_true, hl_pred)

# Compare true pre-post interventions
true_pre_int = df_test.index.values
true_post_int = df_test_intervention.index.values
stats.ks_2samp(true_pre_int, true_post_int)

# Test of half-ogistic fits
hl_true = stats.halflogistic.fit(true_pre_int)
hl_pred = stats.halflogistic.fit(true_post_int)
hl_fit_int = stats.ks_2samp(hl_true, hl_pred)

# Compare predicted pre-post interventions
pred_pre_int = age_prop_pred
pred_post_int = age_prop_int_pred
stats.ks_2samp(pred_pre_int, pred_post_int)

# Test of half-ogistic fits
hl_true = stats.halflogistic.fit(pred_pre_int)
hl_pred = stats.halflogistic.fit(pred_post_int)
hl_fit_int = stats.ks_2samp(hl_true, hl_pred)
