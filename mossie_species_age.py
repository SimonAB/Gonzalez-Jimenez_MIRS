# Import modules

#%%
import os
import glob
import re
import ast
from time import time
from tqdm import tqdm
from collections import Counter


import numpy as np
import pandas as pd

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from random import randint
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import maxabs_scale

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsemble

import matplotlib.pyplot as plt
# %matplotlib inline


import seaborn as sns
sns.set(context="paper",
    style="whitegrid",
    palette="deep",
    font_scale=1.4,
    color_codes=True,
    rc=None)

HOME = os.path.expanduser("~") # just in case we need this later

import warnings
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable oob estimates")

# Set random seed to insure reproducibility
seed = 4

## FUNCTIONS

def plot_confusion_matrix(cm, classes,
                          normalise=True,
                          text=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """

    if normalise:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title="{0} (normalised)".format(title)
            # print("Normalized confusion matrix")
        # else:
            # print('Confusion matrix')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if text:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]),
                                      range(cm.shape[1])):
            plt.text(j, i, "{0:.2f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def cv_report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3%} ± {1:.3%}".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def cv_report_mse(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3} ± {1:.3}".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    # From https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

# import full dataset
df_full = pd.read_table("./mosquitoes_spectra (170623).dat")

df_full.head()
# select species data
df_species = df_full.copy()
df_species.index = df_species["Species"]
df_species = df_species.iloc[:,5:]

# transform species data
df_species[df_species.columns] = StandardScaler().fit_transform(df_species[df_species.columns].as_matrix())

# select real age data
df_real_age = df_full.copy()
df_real_age.index = df_real_age["Real age"]
df_real_age = df_real_age.iloc[:,5:]
# transform data
df_real_age[df_real_age.columns] = StandardScaler().fit_transform(df_real_age[df_real_age.columns].as_matrix())

# select age of AG data
df_real_age_AG = df_full.copy()
df_real_age_AG.index = df_real_age_AG["Real age"]
df_real_age_AG = df_real_age_AG[df_real_age_AG["Species"] == "AG"]
df_real_age_AG = df_real_age_AG.iloc[:,5:]
# transform data
df_real_age_AG[df_real_age_AG.columns] = StandardScaler().fit_transform(df_real_age_AG[df_real_age_AG.columns].as_matrix())
df_real_age_AG.tail()

# select age of AR data
df_real_age_AR = df_full.copy()
df_real_age_AR.index = df_real_age_AR["Real age"]
df_real_age_AR = df_real_age_AR[df_real_age_AR["Species"] == "AR"]
df_real_age_AR = df_real_age_AR.iloc[:,5:]
# transform data
df_real_age_AR[df_real_age_AR.columns] = StandardScaler().fit_transform(df_real_age_AR[df_real_age_AR.columns].as_matrix())
df_real_age_AR.tail()


# select age data
df_age = df_full.copy()
df_age.index = df_age["Age"]
df_age = df_age.iloc[:,5:]
# transform data
df_age[df_age.columns] = StandardScaler().fit_transform(df_age[df_age.columns].as_matrix())

# select age of AG data
df_age_AG = df_full.copy()
df_age_AG.index = df_age_AG["Age"]
df_age_AG = df_age_AG[df_age_AG["Species"] == "AG"]
df_age_AG = df_age_AG.iloc[:,5:]
# transform data
df_age_AG[df_age_AG.columns] = StandardScaler().fit_transform(df_age_AG[df_age_AG.columns].as_matrix())
df_age_AG.tail()

# select age of AR data
df_age_AR = df_full.copy()
df_age_AR.index = df_age_AR["Age"]
df_age_AR = df_age_AR[df_age_AR["Species"] == "AR"]
df_age_AR = df_age_AR.iloc[:,5:]
# transform data
df_age_AR[df_age_AR.columns] = StandardScaler().fit_transform(df_age_AR[df_age_AR.columns].as_matrix())
df_age_AR.tail()


# feed
df_all_food = df_full.copy()
df_all_food.index = df_all_food["Status"]
df_all_food = df_all_food.iloc[:,5:]

df_sp_food = df_full.copy()
df_sp_food.index = df_sp_food["Age"]
df_BF = df_sp_food[df_sp_food["Status"] == "BF"].iloc[:,5:]
df_SF = df_sp_food[df_sp_food["Status"] == "SF"].iloc[:,5:]
df_GR = df_sp_food[df_sp_food["Status"] == "GR"].iloc[:,5:]

# import species+age data
df_species_age = df_full.copy()
df_species_age.index = df_species_age["Species"] + "_" + df_species_age["Age"]

df_species_age = df_species_age.iloc[:,5:]

# transform data
df_species_age[df_species_age.columns] = StandardScaler().fit_transform(df_species_age[df_species_age.columns].as_matrix())

# import field test dataset
# df_fieldtest = pd.read_table("field_mosquitoes_spectra.dat")
# df_fieldtest = df_fieldtest[df_fieldtest["Status"] == "BA"]

# df_fieldtest_species_age = df_fieldtest.copy()
# df_fieldtest_species_age.index = df_fieldtest_species_age["Species"] + "_" + df_fieldtest_species_age["Age"]
# df_fieldtest_species_age = df_fieldtest_species_age.iloc[:,5:]
# df_fieldtest_species_age[df_fieldtest_species_age.columns] = StandardScaler().fit_transform(df_fieldtest_species_age[df_fieldtest_species_age.columns].as_matrix())
#
#
# df_fieldtest_age = df_fieldtest.copy()
# df_fieldtest_age.index = df_fieldtest_age["Age"]
# df_fieldtest_age = df_fieldtest_age.iloc[:,5:]
# df_fieldtest_age[df_fieldtest_age.columns] = StandardScaler().fit_transform(df_fieldtest_age[df_fieldtest_age.columns].as_matrix())
#
# df_fieldtest_species = df_fieldtest.copy()
# df_fieldtest_species.index = df_fieldtest_species["Species"]
# df_fieldtest_species = df_fieldtest_species.iloc[:,5:]
# df_fieldtest_species[df_fieldtest_species.columns] = StandardScaler().fit_transform(df_fieldtest_species[df_fieldtest_species.columns].as_matrix())







#########################################
# PREDICTING SPECIES
#########################################

# Spot-checking classification approaches
#%%

# determine size of data to use
X = df_species.astype(float)
y = df_species.index

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=34)
X_resampled, y_resampled = rus.fit_sample(X, y)

X_resampled_df, X_resampled_df.index, X_resampled_df.columns = pd.DataFrame(X_resampled), y_resampled, X.columns
y_resampled_df = X_resampled_df.index

# cross-val settings

validation_size = 0.3
num_splits = 10

models = []
models.append(("KNN", KNeighborsClassifier()))
models.append(("LR", LogisticRegressionCV()))
# models.append(("SGD", SGDClassifier()))
models.append(("SVM", SVC()))
models.append(("NB", GaussianNB()))
# models.append(("LDA", LinearDiscriminantAnalysis()))
# models.append(("CART", DecisionTreeClassifier()))
models.append(("RF", RandomForestClassifier()))
# models.append(("ET", ExtraTreeClassifier()))
models.append(("XGB", XGBClassifier()))

# generate results for each model in turn
results = []
names = []
scoring = "accuracy"

for name, model in models:
    #    kfold = KFold(n=num_instances, n_splits=num_splits, random_state=seed)
    # kfold = StratifiedKFold(y, n_splits=num_splits, shuffle=True,
    # random_state=seed) # stratifiedKFold fails with ValueError: array must
    # not contain infs or NaNs
    sss = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=validation_size, random_state=seed)
    sss.split(X_resampled_df, y_resampled_df)
    cv_results = cross_val_score(model, X_resampled_df, y_resampled_df, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)



# plot
#%%
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Predicting Mosquito Species", weight="bold")
plt.xticks(rotation=30)
plt.ylabel("Accuracy (median, quartiles, range)")
plt.savefig("./plots/spot_check_species_rus.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_species_rus.png", bbox_inches="tight")




# Optimising XGBoost

#%%

# Parameter search

# features & labels
X = df_species.values
y = df_species.index
Counter(y)

# cross validation
validation_size = 0.3
num_splits = 5
num_repeats = 5
num_rounds = 5
scoring = "accuracy"

# preparing model
model = XGBClassifier(nthread=1, seed=seed)

# Grid search paramater space
colsample_bytree = [0.1, 0.3, 0.5, 0.8, 1]
learning_rate = [0.001, 0.01, 0.1]
max_depth = [6, 8, 10]
min_child_weight = [1, 3, 5, 7]
n_estimators = [50, 100, 300, 500]

# Mini Grid search paramater space
# colsample_bytree = [0.1, 1]
# learning_rate = [0.001, 0.01]
# max_depth = [6, 8]
# min_child_weight = [1, 3]
# n_estimators = [100, 300]

parameters = {"colsample_bytree": colsample_bytree,
              "learning_rate": learning_rate,
              "min_child_weight": min_child_weight,
              "n_estimators": n_estimators,
              "max_depth": max_depth}


# repeated random stratified splitting of dataset
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame() # model parameters and global accuracy score
rskf_per_class_results = [] # per class accuracy scores
rskf_orphan_preds = pd.DataFrame() # orphan predictions
start = time()

for round in range(num_rounds):
    seed=np.random.randint(0, 81470108)

    # under-sample over-represented classes
    rus = RandomUnderSampler(random_state=seed)
    X_resampled, y_resampled = rus.fit_sample(X, y) #produces numpy arrays

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # GRID SEARCH
        # # grid search on each iteration
        # start = time()
        # gsCV = GridSearchCV(estimator=classifier, param_grid=parameters,
        #                     scoring=scoring, cv=rskf, n_jobs=-1, verbose=1)
        # CV_result = gsCV.fit(X_train, y_train)
        # best_model = model(**CV_result.best_params_)

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(verbose=1,
            estimator=model, param_distributions=parameters, n_iter=n_iter_search, cv=rskf, n_jobs=-1)
        rsCV_result = rsCV.fit(X_train, y_train)

        best_model = XGBClassifier(nthread=1, seed=seed, **rsCV_result.best_params_)

        #fit model
        best_model.fit(X_train, y_train)

        #test model
        y_pred = best_model.predict(np.delete(X_resampled, train_index, axis=0))
        y_test = np.delete(y_resampled, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)
        local_feat_impces = pd.DataFrame(best_model.feature_importances_, index=df_species.columns).sort_values(by=0, ascending=False)

        local_rskf_results = pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("params",str(rsCV_result.best_params_)), ("seed", best_model.seed), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report), ("Feature importances", local_feat_impces.to_dict())]).T


        local_rskf_results.columns=local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv("./results/xgb_sp_repeatedCV_record.csv", index=False)
rskf_results = pd.read_csv("./results/xgb_sp_repeatedCV_record.csv")


# Accuracy distribution
xgb_sp_acc_distrib = rskf_results["Accuracy"]
xgb_sp_acc_distrib.columns=["Accuracy"]
xgb_sp_acc_distrib.to_csv("./results/xgb_sp_acc_distrib.csv", header=True, index=False)
xgb_sp_acc_distrib = pd.read_csv("./results/xgb_sp_acc_distrib.csv")
xgb_sp_acc_distrib = np.round(100*xgb_sp_acc_distrib)

plt.figure(figsize=(2.25,3))
sns.distplot(xgb_sp_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/xgb_sp_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_sp_acc_distrib.png", bbox_inches="tight")

class_names = y.sort_values().unique()
xgb_sp_per_class_acc_distrib = pd.DataFrame(rskf_per_class_results, columns=class_names)
xgb_sp_per_class_acc_distrib.dropna().to_csv("./results/xgb_sp_per_class_acc_distrib.csv")
xgb_sp_per_class_acc_distrib = pd.read_csv("./results/xgb_sp_per_class_acc_distrib.csv", index_col=0)
xgb_sp_per_class_acc_distrib = np.round(100*xgb_sp_per_class_acc_distrib)
xgb_sp_per_class_acc_distrib_describe = xgb_sp_per_class_acc_distrib.describe()
xgb_sp_per_class_acc_distrib_describe.to_csv("./results/xgb_sp_per_class_acc_distrib.csv")

xgb_sp_per_class_acc_distrib = pd.melt(xgb_sp_per_class_acc_distrib, var_name="Reservoir")

plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="Reservoir", y="value", cut=0, data=xgb_sp_per_class_acc_distrib)
sns.despine(left=True)
# plt.xticks(rotation=45, ha="right")
plt.xlabel("Species")
plt.ylabel("Prediction accuracy\n ({0:.2f} ± {1:.2f})".format(xgb_sp_per_class_acc_distrib["value"].mean(),xgb_sp_per_class_acc_distrib["value"].sem()), weight="bold")
plt.savefig("./plots/xgb_sp_per_class_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_sp_per_class_acc_distrib.png", bbox_inches="tight")



# Feature Importances

#%%
## make this into bar with error bars across all best models

rskf_results = pd.read_csv("./results/xgb_sp_repeatedCV_record.csv")

# All feat imp
all_featimp = pd.DataFrame(ast.literal_eval(rskf_results["Feature importances"][0]))

for featimp in rskf_results["Feature importances"][1:]:
    featimp = pd.DataFrame(ast.literal_eval(featimp))
    all_featimp = all_featimp.merge(featimp, left_index=True, right_index=True)

all_featimp["mean"] = all_featimp.mean(axis=1)
all_featimp["sem"] = all_featimp.sem(axis=1)
all_featimp.sort_values(by="mean", inplace=True)

featimp_global_mean = all_featimp["mean"].mean()
featimp_global_sem = all_featimp["mean"].sem()


fig = all_featimp["mean"][-8:].plot(figsize=(2.2, 3),
                                    kind="barh",
                                    legend=False,
                                    xerr=all_featimp["sem"],
                                    ecolor='k')
plt.xlabel("Feature importance")
plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# fig.set_yticklabels([
#     "CD8 T cells (# in thymus)",
#     "MEP cells (# in bone marrow)",
#     "Erythroid Ter119 cells (# in bone marrow)",
#     "Memory CD4 T cells (% PBMC)",
#     "Memory T cells (% PBMC)",
#     "Memory T cells (% spleen)",
#     "Naive T cells (% spleen)",
#     "Naive CD4 T cells (% spleen)",
#     ])

plt.savefig("./plots/xgb_sp_feat_imp.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_sp_feat_imp.png", bbox_inches="tight")




#########################################
# PREDICTING AGE OF AG ONLY
#########################################

# Spot-checking classification approaches
#%%

# determine size of data to use
X = df_age_AG.astype(float)
y = df_age_AG.index

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=34)
X_resampled, y_resampled = rus.fit_sample(X, y)

# cross-val settings

validation_size = 0.3
num_splits = 10

models = []
models.append(("KNN", KNeighborsClassifier()))
# models.append(("SGD", SGDClassifier()))
# models.append(("LDA", LinearDiscriminantAnalysis(random_state=seed)))
models.append(("LR", LogisticRegressionCV()))
# models.append(("CART", DecisionTreeClassifier()))
models.append(("SVM", SVC()))
models.append(("NB", GaussianNB()))
models.append(("RF", RandomForestClassifier()))
# models.append(("ET", ExtraTreeClassifier()))
models.append(("XGB", XGBClassifier()))

# generate results for each model in turn
results = []
names = []
scoring = "accuracy"

for name, model in models:
    #    kfold = KFold(n=num_instances, n_splits=num_splits, random_state=seed)
    # kfold = StratifiedKFold(y, n_splits=num_splits, shuffle=True,
    # random_state=seed) # stratifiedKFold fails with ValueError: array must
    # not contain infs or NaNs
    sss = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=validation_size, random_state=seed)
    sss.split(X_resampled, y_resampled)
    cv_results = cross_val_score(model, X_resampled, y_resampled, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)


# plot

#%%
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Predicting AG age", weight="bold")
plt.xticks(rotation=30)
plt.ylabel("Accuracy (median, quartiles, range)")
plt.savefig("./plots/spot_check_age_rus_AG.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_age_rus_AG.png", bbox_inches="tight")




## Logistic regression

#%%
# features & labels
X = df_age_AG.values
y = df_age_AG.index

# cross validation

validation_size = 0.3
num_splits = 5
num_repeats = 10
num_rounds = 10
scoring = "accuracy"

# preparing model
classifier = LogisticRegressionCV(Cs=10,
                            fit_intercept=True,
                            cv=10,
                            dual=False,
                            penalty="l2",
                            scoring="accuracy",
                            solver="lbfgs",
                            tol=0.0001,
                            max_iter=100,
                            class_weight="balanced",
                            n_jobs=-1,
                            verbose=0,
                            refit=True,
                            intercept_scaling=1.0,
                            multi_class="ovr",
                            random_state=seed)

# repeated random stratified splitting of dataset
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame() # model parameters and global accuracy score
rskf_per_class_results = [] # per class accuracy scores
rkf_scores = pd.DataFrame(columns=["age", "scores mean", "scores sem"]).set_index("age")

start = time()

for round in range(num_rounds):
    seed=np.random.randint(0, 81470108)

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit model
        classifier.fit(X_train, y_train)

        #test model
        y_pred = classifier.predict(np.delete(X, train_index, axis=0))
        y_test = np.delete(y, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)
        local_scores = pd.DataFrame.from_records([classifier.scores_])

        scores_table = pd.DataFrame({"scores":pd.Series(classifier.scores_), "age":df_age_AG.index.unique()}).set_index("age")

        # combine score outputs
        rkf_scores = pd.merge(rkf_scores, scores_table, left_index=True, right_index=True, how='outer')

        local_rskf_results = pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report), ("Scores", local_scores)]).T

        local_rskf_results.columns=local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv("./results/lr_AG_age_repeatedCV_record.csv", index=False)

# rename columns to avoid duplicates
nameslist=list(range(1,rkf_scores.shape[1]-1))
rkf_scores.columns = rkf_scores.columns[:2].tolist() + nameslist

# calculate mean and sem of scores
rownum=0
for rowid in rkf_scores.index:
    rkf_scores.at[rowid, "scores mean"] = np.mean(np.mean(rkf_scores.iloc[rownum,3:]))
    rkf_scores.at[rowid, "scores sem"] = np.mean(stats.sem(np.mean(rkf_scores.iloc[rownum,3:])))
    rownum += 1


rkf_scores.dropna(axis=1).to_csv("./results/lr_AG_age_rkf_scores.csv", index=False)

rskf_results = pd.read_csv("./results/lr_AG_age_repeatedCV_record.csv")
rkf_scores = pd.read_csv("./results/lr_AG_age_rkf_scores.csv")
rkf_scores.index=["1", "3", "5", "7", "old"]


# Accuracy distribution
lr_AG_age_acc_distrib = rskf_results["Accuracy"]
lr_AG_age_acc_distrib.columns=["Accuracy"]
lr_AG_age_acc_distrib.to_csv("./results/lr_AG_age_acc_distrib.csv", header=True, index=False)
lr_AG_age_acc_distrib = pd.read_csv("./results/lr_AG_age_acc_distrib.csv")
lr_AG_age_acc_distrib = np.round(100*lr_AG_age_acc_distrib)

plt.figure(figsize=(2.25,3))
sns.distplot(lr_AG_age_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/lr_AG_age_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_AG_age_acc_distrib.png", bbox_inches="tight")

# Per class accuracy
class_names = y.sort_values().unique()
lr_AG_age_per_class_acc_distrib = pd.DataFrame(rskf_per_class_results, columns=class_names)
lr_AG_age_per_class_acc_distrib.dropna().to_csv("./results/lr_AG_age_per_class_acc_distrib.csv")
lr_AG_age_per_class_acc_distrib = pd.read_csv("./results/lr_AG_age_per_class_acc_distrib.csv", index_col=0)
lr_AG_age_per_class_acc_distrib = np.round(100*lr_AG_age_per_class_acc_distrib)
lr_AG_age_per_class_acc_distrib_describe = lr_AG_age_per_class_acc_distrib.describe()
lr_AG_age_per_class_acc_distrib_describe.to_csv("./results/lr_AG_age_per_class_acc_distrib.csv")

lr_AG_age_per_class_acc_distrib = pd.melt(lr_AG_age_per_class_acc_distrib, var_name="AG age")
lr_AG_age_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="AG age", y="value", cut=0, data=lr_AG_age_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/lr_AG_age_per_class_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_AG_age_per_class_acc_distrib.png", bbox_inches="tight")


# LR coeffients

#%%
# ....





#########################################
# PREDICTING AGE OF AR ONLY
#########################################

# Spot-checking classification approaches
#%%

# determine size of data to use
X = df_age_AR.astype(float)
y = df_age_AR.index

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=34)
X_resampled, y_resampled = rus.fit_sample(X, y)

# cross-val settings

validation_size = 0.3
num_splits = 10

models = []
models.append(("LR", LogisticRegression(random_state=seed)))
models.append(("SGD", SGDClassifier(random_state=seed)))
# models.append(("LDA", LinearDiscriminantAnalysis(random_state=seed)))
models.append(("KNN", KNeighborsClassifier(random_state=seed)))
models.append(("CART", DecisionTreeClassifier(random_state=seed)))
models.append(("RF", RandomForestClassifier(random_state=seed)))
models.append(("ET", ExtraTreeClassifier(random_state=seed)))
models.append(("XGB", XGBClassifier(random_state=seed)))
models.append(("NB", GaussianNB(random_state=seed)))
models.append(("SVM", SVC(random_state=seed)))

# generate results for each model in turn
results = []
names = []
scoring = "accuracy"

for name, model in models:
    #    kfold = KFold(n=num_instances, n_splits=num_splits, random_state=seed)
    # kfold = StratifiedKFold(y, n_splits=num_splits, shuffle=True,
    # random_state=seed) # stratifiedKFold fails with ValueError: array must
    # not contain infs or NaNs
    sss = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=validation_size, random_state=seed)
    sss.split(X_resampled, y_resampled)
    cv_results = cross_val_score(model, X_resampled, y_resampled, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)


# plot

#%%
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=30)
plt.ylabel("Accuracy (median, quartiles, range)")
plt.savefig("./plots/spot_check_age_rus_AR.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_age_rus_AR.png", bbox_inches="tight")



# BIG LOOP

#%%

# Parameter search

# features & labels
X = df_age_AR.values
y = df_age_AR.index

# cross validation

validation_size = 0.3
num_splits = 5
num_repeats = 5
num_rounds = 5
scoring = "accuracy"

# preparing model
model = XGBClassifier(nthread=1, seed=seed)

# Grid search paramater space
colsample_bytree = [0.1, 0.3, 0.5, 0.8, 1]
learning_rate = [0.001, 0.01, 0.1]
max_depth = [6, 8, 10]
min_child_weight = [1, 3, 5, 7]
n_estimators = [50, 100, 300, 500]

# Mini Grid search paramater space
# colsample_bytree = [0.1, 1]
# learning_rate = [0.001, 0.01]
# max_depth = [6, 8]
# min_child_weight = [1, 3]
# n_estimators = [100, 300]

parameters = {"colsample_bytree": colsample_bytree,
              "learning_rate": learning_rate,
              "min_child_weight": min_child_weight,
              "n_estimators": n_estimators,
              "max_depth": max_depth}


# repeated random stratified splitting of dataset
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame() # model parameters and global accuracy score
rskf_per_class_results = [] # per class accuracy scores
rskf_orphan_preds = pd.DataFrame() # orphan predictions
start = time()

for round in range(num_rounds):
    seed=np.random.randint(0, 81470108)

    # under-sample over-represented classes
    rus = RandomUnderSampler(random_state=seed, ratio="auto")
    X_resampled, y_resampled = rus.fit_sample(X, y) #produces numpy arrays

    for train_index, test_index in rskf.split(X_resampled, y_resampled):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # GRID SEARCH
        # # grid search on each iteration
        # start = time()
        # gsCV = GridSearchCV(estimator=classifier, param_grid=parameters,
        #                     scoring=scoring, cv=rskf, n_jobs=-1, verbose=1)
        # CV_result = gsCV.fit(X_train, y_train)
        # best_model = model(**CV_result.best_params_)

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(verbose=1,
            estimator=model, param_distributions=parameters, n_iter=n_iter_search, cv=rskf, n_jobs=-1)
        rsCV_result = rsCV.fit(X_train, y_train)

        best_model = XGBClassifier(nthread=1, seed=seed, **rsCV_result.best_params_)

        #fit model
        best_model.fit(X_train, y_train)

        #test model
        y_pred = best_model.predict(np.delete(X_resampled, train_index, axis=0))
        y_test = np.delete(y_resampled, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)

        local_rskf_results = pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("params",str(rsCV_result.best_params_)), ("seed", best_model.seed), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report)]).T

        local_rskf_results.columns=local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv("./results/xgb_AR_age_repeatedCV_record.csv", index=False)
rskf_results = pd.read_csv("./results/xgb_AR_age_repeatedCV_record.csv")


# Accuracy distribution
xgb_AR_age_acc_distrib = rskf_results["Accuracy"]
xgb_AR_age_acc_distrib.columns=["Accuracy"]
xgb_AR_age_acc_distrib.to_csv("./results/xgb_AR_age_acc_distrib.csv", header=True, index=False)
xgb_AR_age_acc_distrib = pd.read_csv("./results/xgb_AR_age_acc_distrib.csv")
xgb_AR_age_acc_distrib = np.round(100*xgb_AR_age_acc_distrib)

plt.figure(figsize=(2.25,3))
sns.distplot(xgb_AR_age_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/xgb_AR_age_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_AR_age_acc_distrib.png", bbox_inches="tight")

class_names = y.sort_values().unique()
xgb_AR_age_per_class_acc_distrib = pd.DataFrame(rskf_per_class_results, columns=class_names)
xgb_AR_age_per_class_acc_distrib.dropna().to_csv("./results/xgb_AR_age_per_class_acc_distrib.csv")
xgb_AR_age_per_class_acc_distrib = pd.read_csv("./results/xgb_AR_age_per_class_acc_distrib.csv", index_col=0)
xgb_AR_age_per_class_acc_distrib = np.round(100*xgb_AR_age_per_class_acc_distrib)
xgb_AR_age_per_class_acc_distrib_describe = xgb_AR_age_per_class_acc_distrib.describe()
xgb_AR_age_per_class_acc_distrib_describe.to_csv("./results/xgb_AR_age_per_class_acc_distrib.csv")

xgb_AR_age_per_class_acc_distrib = pd.melt(xgb_AR_age_per_class_acc_distrib, var_name="AG age")
xgb_AR_age_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="AG age", y="value", cut=0, data=xgb_AR_age_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/xgb_AR_age_per_class_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_AR_age_per_class_acc_distrib.png", bbox_inches="tight")




######################################################################
# PREDICTING AGE OF BOTH SPECIES (all feeds)
######################################################################

# Spot-checking classification approaches
#%%

# determine size of data to use
X = df_age.astype(float)
y = df_age.index
Counter(y)

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=seed)
X_resampled, y_resampled = rus.fit_sample(X, y)

X_resampled_df, X_resampled_df.index, X_resampled_df.columns = pd.DataFrame(X_resampled), y_resampled, X.columns
y_resampled_df = X_resampled_df.index

# cross-val settings
validation_size = 0.3
num_splits = 10

models = []
models.append(("LR", LogisticRegression()))
models.append(("SGD", SGDClassifier()))
# models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier(random_state=seed)))
models.append(("RF", RandomForestClassifier(random_state=seed)))
models.append(("ET", ExtraTreeClassifier(random_state=seed)))
models.append(("XGB", XGBClassifier(random_state=seed)))
models.append(("NB", GaussianNB(random_state=seed)))
models.append(("SVM", SVC(random_state=seed)))

# generate results for each model in turn
results = []
names = []
scoring = "accuracy"

for name, model in models:
    #    kfold = KFold(n=num_instances, n_splits=num_splits, random_state=seed)
    # kfold = StratifiedKFold(y, n_splits=num_splits, shuffle=True,
    # random_state=seed) # stratifiedKFold fails with ValueError: array must
    # not contain infs or NaNs
    sss = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=validation_size, random_state=seed)
    sss.split(X_resampled_df, y_resampled_df)
    cv_results = cross_val_score(model, X, y, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)


# plot

#%%
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=30)
plt.ylabel("Accuracy (median, quartiles, range)")
plt.savefig("./plots/spot_check_age_rus_AGandAR.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_age_rus_AGandAR.png", bbox_inches="tight")



# BIG LOOP

#%%

# Parameter search

# features & labels
X = df_age.values
y = df_age.index

# cross validation

validation_size = 0.3
num_splits = 5
num_repeats = 5
num_rounds = 5
scoring = "accuracy"

# preparing model
model = XGBClassifier(nthread=1, seed=seed)

# Grid search paramater space
colsample_bytree = [0.1, 0.3, 0.5, 0.8, 1]
learning_rate = [0.001, 0.01, 0.1]
max_depth = [6, 8, 10]
min_child_weight = [1, 3, 5, 7]
n_estimators = [50, 100, 300, 500]

# Mini Grid search paramater space
# colsample_bytree = [0.1, 1]
# learning_rate = [0.001, 0.01]
# max_depth = [6, 8]
# min_child_weight = [1, 3]
# n_estimators = [100, 300]

parameters = {"colsample_bytree": colsample_bytree,
              "learning_rate": learning_rate,
              "min_child_weight": min_child_weight,
              "n_estimators": n_estimators,
              "max_depth": max_depth}


# repeated random stratified splitting of dataset
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame() # model parameters and global accuracy score
rskf_per_class_results = [] # per class accuracy scores
rskf_orphan_preds = pd.DataFrame() # orphan predictions
start = time()

for round in range(num_rounds):
    seed=np.random.randint(0, 81470108)

    # under-sample over-represented classes
    rus = RandomUnderSampler(random_state=seed, ratio="auto")
    X_resampled, y_resampled = rus.fit_sample(X, y) #produces numpy arrays

    for train_index, test_index in rskf.split(X_resampled, y_resampled):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # GRID SEARCH
        # # grid search on each iteration
        # start = time()
        # gsCV = GridSearchCV(estimator=classifier, param_grid=parameters,
        #                     scoring=scoring, cv=rskf, n_jobs=-1, verbose=1)
        # CV_result = gsCV.fit(X_train, y_train)
        # best_model = model(**CV_result.best_params_)

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(verbose=1,
            estimator=model, param_distributions=parameters, n_iter=n_iter_search, cv=rskf, n_jobs=-1)
        rsCV_result = rsCV.fit(X_train, y_train)

        best_model = XGBClassifier(nthread=1, seed=seed, **rsCV_result.best_params_)

        #fit model
        best_model.fit(X_train, y_train)

        #test model
        y_pred = best_model.predict(np.delete(X_resampled, train_index, axis=0))
        y_test = np.delete(y_resampled, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)

        local_rskf_results = pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("params",str(rsCV_result.best_params_)), ("seed", best_model.seed), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report)]).T

        local_rskf_results.columns=local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv("./results/xgb_age_repeatedCV_record.csv", index=False)
rskf_results = pd.read_csv("./results/xgb_age_repeatedCV_record.csv")


# Accuracy distribution
xgb_age_acc_distrib = rskf_results["Accuracy"]
xgb_age_acc_distrib.columns=["Accuracy"]
xgb_age_acc_distrib.to_csv("./results/xgb_age_acc_distrib.csv", header=True, index=False)
xgb_age_acc_distrib = pd.read_csv("./results/xgb_age_acc_distrib.csv")
xgb_age_acc_distrib = np.round(100*xgb_age_acc_distrib)

plt.figure(figsize=(2.25,3))
sns.distplot(xgb_age_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/xgb_age_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_age_acc_distrib.png", bbox_inches="tight")

class_names = y.sort_values().unique()
xgb_age_per_class_acc_distrib = pd.DataFrame(rskf_per_class_results, columns=class_names)
xgb_age_per_class_acc_distrib.dropna().to_csv("./results/xgb_age_per_class_acc_distrib.csv")
xgb_age_per_class_acc_distrib = pd.read_csv("./results/xgb_age_per_class_acc_distrib.csv", index_col=0)
xgb_age_per_class_acc_distrib = np.round(100*xgb_age_per_class_acc_distrib)
xgb_age_per_class_acc_distrib_describe = xgb_age_per_class_acc_distrib.describe()
xgb_age_per_class_acc_distrib_describe.to_csv("./results/xgb_age_per_class_acc_distrib.csv")

xgb_age_per_class_acc_distrib = pd.melt(xgb_age_per_class_acc_distrib, var_name="Age")
xgb_age_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="Age", y="value", cut=0, data=xgb_age_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/xgb_age_per_class_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_age_per_class_acc_distrib.png", bbox_inches="tight")





#########################################
# PREDICTING AGE AND SPECIES
#########################################

# Spot-checking classification approaches
#%%

# determine size of data to use
X = df_species_age.astype(float)
y = df_species_age.index

Counter(y)

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=4)
# rus = RandomUnderSampler(random_state=4, ratio={"AG_1":55, "AG_3":141, "AG_5":115, "AG_7":121, "AG_old":130, "AR_1":66, "AR_3":97, "AR_5":96, "AR_7":155, "AR_old":130})
X_resampled, y_resampled = rus.fit_sample(X, y)

X_resampled_df, X_resampled_df.index, X_resampled_df.columns = pd.DataFrame(X_resampled), y_resampled, X.columns
y_resampled_df = X_resampled_df.index
#
# Counter(y_resampled_df)

# cross-val settings

validation_size = 0.3
num_splits = 10

models = []
models.append(("LR", LogisticRegression(random_state=seed)))
models.append(("SGD", SGDClassifier(random_state=seed)))
# models.append(("LDA", LinearDiscriminantAnalysis(random_state=seed)))
models.append(("KNN", KNeighborsClassifier(random_state=seed)))
models.append(("CART", DecisionTreeClassifier(random_state=seed)))
models.append(("RF", RandomForestClassifier(random_state=seed)))
models.append(("ET", ExtraTreeClassifier(random_state=seed)))
models.append(("XGB", XGBClassifier(random_state=seed)))
models.append(("NB", GaussianNB(random_state=seed)))
models.append(("SVM", SVC(random_state=seed)))

# generate results for each model in turn
results = []
names = []
scoring = "accuracy"

for name, model in models:
    #    kfold = KFold(n=num_instances, n_splits=num_splits, random_state=seed)
    # kfold = StratifiedKFold(y, n_splits=num_splits, shuffle=True,
    # random_state=seed) # stratifiedKFold fails with ValueError: array must
    # not contain infs or NaNs
    sss = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=validation_size, random_state=seed)
    sss.split(X, y)
    cv_results = cross_val_score(model, X, y, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)



# plot
#%%
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=30)
plt.ylabel("Accuracy (median, quartiles, range)")
plt.savefig("./plots/spot_check_species_age_rus.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_species_age_rus.png", bbox_inches="tight")




# Big loop

#%%
# determine size of data to use
X = df_species_age.values
y = df_species_age.index

# cross validation

validation_size = 0.3
num_splits = 5
num_repeats = 5
num_rounds = 5
scoring = "accuracy"

# preparing model
model = XGBClassifier(nthread=1, seed=seed)

# Grid search paramater space
colsample_bytree = [0.1, 0.3, 0.5, 0.8, 1]
learning_rate = [0.001, 0.01, 0.1]
max_depth = [6, 8, 10]
min_child_weight = [1, 3, 5, 7]
n_estimators = [50, 100, 300, 500]

# Mini Grid search paramater space
# colsample_bytree = [0.1, 1]
# learning_rate = [0.001, 0.01]
# max_depth = [6, 8]
# min_child_weight = [1, 3]
# n_estimators = [100, 300]

parameters = {"colsample_bytree": colsample_bytree,
              "learning_rate": learning_rate,
              "min_child_weight": min_child_weight,
              "n_estimators": n_estimators,
              "max_depth": max_depth}


# repeated random stratified splitting of dataset
rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame() # model parameters and global accuracy score
rskf_per_class_results = [] # per class accuracy scores
rskf_orphan_preds = pd.DataFrame() # orphan predictions
start = time()

for round in range(num_rounds):
    seed=np.random.randint(0, 81470108)

    # under-sample over-represented classes
    rus = RandomUnderSampler(random_state=seed)
    X_resampled, y_resampled = rus.fit_sample(X, y) #produces numpy arrays

    for train_index, test_index in rskf.split(X_resampled, y_resampled):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # GRID SEARCH
        # # grid search on each iteration
        # start = time()
        # gsCV = GridSearchCV(estimator=classifier, param_grid=parameters,
        #                     scoring=scoring, cv=rskf, n_jobs=-1, verbose=1)
        # CV_result = gsCV.fit(X_train, y_train)
        # best_model = model(**CV_result.best_params_)

        # RANDOMSED GRID SEARCH
        n_iter_search = 10
        rsCV = RandomizedSearchCV(verbose=1,
            estimator=model, param_distributions=parameters, n_iter=n_iter_search, cv=rskf, n_jobs=-1)
        rsCV_result = rsCV.fit(X_train, y_train)

        best_model = XGBClassifier(nthread=1, seed=seed, **rsCV_result.best_params_)

        #fit model
        best_model.fit(X_train, y_train)

        #test model
        y_pred = best_model.predict(np.delete(X_resampled, train_index, axis=0))
        y_test = np.delete(y_resampled, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)

        local_rskf_results = pd.DataFrame([("Accuracy",accuracy_score(y_test, y_pred)), ("params",str(rsCV_result.best_params_)), ("seed", best_model.seed), ("TRAIN",str(train_index)), ("TEST",str(test_index)), ("CM", local_cm), ("Classification report", local_report)]).T

        local_rskf_results.columns=local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv("./results/xgb_repeatedCV_record.csv", index=False)
rskf_results = pd.read_csv("./results/xgb_repeatedCV_record.csv")


# Accuracy distribution
xgb_acc_distrib = rskf_results["Accuracy"]
xgb_acc_distrib.columns=["Accuracy"]
xgb_acc_distrib.to_csv("./results/xgb_acc_distrib.csv", header=True, index=False)
xgb_acc_distrib = pd.read_csv("./results/xgb_acc_distrib.csv")
xgb_acc_distrib = np.round(100*xgb_acc_distrib)

plt.figure(figsize=(2.25,3))
sns.distplot(xgb_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/xgb_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_acc_distrib.png", bbox_inches="tight")

class_names = y.sort_values().unique()
xgb_per_class_acc_distrib = pd.DataFrame(rskf_per_class_results, columns=class_names)
xgb_per_class_acc_distrib.dropna().to_csv("./results/xgb_per_class_acc_distrib.csv")
xgb_per_class_acc_distrib = pd.read_csv("./results/xgb_per_class_acc_distrib.csv", index_col=0)
xgb_per_class_acc_distrib = np.round(100*xgb_per_class_acc_distrib)
xgb_per_class_acc_distrib_describe = xgb_per_class_acc_distrib.describe()
xgb_per_class_acc_distrib_describe.to_csv("./results/xgb_per_class_acc_distrib.csv")

xgb_per_class_acc_distrib = pd.melt(xgb_per_class_acc_distrib, var_name="Species_Age")
xgb_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="Species_Age", y="value", cut=0, data=xgb_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/xgb_per_class_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_per_class_acc_distrib.png", bbox_inches="tight")




######################################################################
# PREDICTING REAL AGE OF BOTH SPECIES (all feeds)
######################################################################


# Spot-checking

#%%
X = df_real_age
y = df_real_age.index

# cross validation
seed = 4
validation_size = 0.25
num_splits = 10
num_repeats = 10
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=num_splits, random_state=seed)

models = []
models.append(("LR", LinearRegression()))
models.append(("EN", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=num_splits, random_state=seed)))
# models.append(("SGD", SGDRegressor()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNNR", KNeighborsRegressor()))
# models.append(("CART", DecisionTreeRegressor(random_state=seed)))
models.append(("RF", RandomForestRegressor(random_state=seed)))
# models.append(("ET", ExtraTreeRegressor(random_state=seed)))
models.append(("XGB", XGBRegressor(random_state=seed)))
# models.append(("kNR", KNeighborsRegressor))
# models.append(("ADA", AdaBoostRegressor))

# generate results for each model in turn
results = []
names = []
scoring = "neg_mean_squared_error"

for name, model in models:
    kfold = KFold(n_splits=num_splits, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2} ± {2:.2}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)


# plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Worm burden")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_real_age.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_real_age.png", bbox_inches="tight")


## search best parameters
#%%
# load data
X = df_real_age.values
y = df_real_age.index

# cross validation
seed = 4
validation_size = 0.25
num_splits = 10
num_repeats = 10
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=num_splits, random_state=seed)

# base algorithm settings
regressor = XGBRegressor(objective="reg:linear", nthread=1)

# define hyperparameter space to test
## for troubleshooting:
# max_depth = [2, 4]
# min_child_weight = [3, 7]
# learning_rate = [0.01, 0.1]
# n_estimators = [5, 10, 20]

# the real deal:
max_depth = [2, 4, 6, 8] #4?
min_child_weight = [1, 3, 5, 7]
n_estimators = [5, 10, 15, 20, 25, 30]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2] #0.1?
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1] #0.2?


parameters = {"max_depth": max_depth,
              "min_child_weight": min_child_weight,
              "learning_rate": learning_rate,
              "n_estimators": n_estimators,
              "colsample_bytree":colsample_bytree}

# repeated random stratified splitting of dataset
rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rkf_results = pd.DataFrame() # model parameters and global accuracy score
rkf_per_class_results = [] # per class accuracy scores

for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # GRID SEARCH
    gsCV = GridSearchCV(estimator=regressor, param_grid=parameters, scoring=scoring, cv=kfold, n_jobs=-1)
    CV_result = gsCV.fit(X_train, y_train)
    best_model = XGBRegressor(objective="reg:linear", nthread=1, **CV_result.best_params_)

    #fit model
    best_model.fit(X_train, y_train)

    #test model
    y_pred = best_model.predict(X_test)

    local_feat_impces = pd.DataFrame(best_model.feature_importances_, index=df_real_age.columns).sort_values(by=0, ascending=False)

    local_rkf_results = pd.DataFrame([("Accuracy",mean_squared_error(y_test, y_pred)), ("params",str(CV_result.best_params_)), ("seed", best_model.seed), ("TRAIN",str(train_index)), ("TEST",str(test_index)),  ("Feature importances", local_feat_impces.to_dict())]).T

    # combine outputs
    local_rkf_results.columns=local_rkf_results.iloc[0]
    local_rkf_results = local_rkf_results[1:]
    rkf_results = rkf_results.append(local_rkf_results)


# Results
rkf_results.to_csv("xgb_real_age_repeatedCV_record.csv", index=False)
rkf_results = pd.read_csv("xgb_real_age_repeatedCV_record.csv")


# Accuracy distribution
xgb_acc_distrib = rkf_results["Accuracy"]
xgb_acc_distrib.columns=["Accuracy"]
xgb_acc_distrib.to_csv("xgb_real_age_acc_distrib.csv", header=True, index=False)



# Feature Importances

#%%
rskf_results = pd.read_csv("xgb_real_age_repeatedCV_record.csv")
xgb_acc_distrib = pd.read_csv("xgb_real_age_acc_distrib.csv")

# All feat imp
all_featimp = pd.DataFrame(ast.literal_eval(rskf_results["Feature importances"][0]))

for featimp in rskf_results["Feature importances"][1:]:
    featimp = pd.DataFrame(ast.literal_eval(featimp))
    all_featimp = all_featimp.merge(featimp, left_index=True, right_index=True)

all_featimp["mean"] = all_featimp.mean(axis=1)
all_featimp["sem"] = all_featimp.sem(axis=1)
all_featimp.sort_values(by="mean", inplace=True)

featimp_global_mean = all_featimp["mean"].mean()
featimp_global_sem = all_featimp["mean"].sem()


fig = all_featimp["mean"][-13:].plot(figsize=(2.2, 4),
                                    kind="barh",
                                    legend=False,
                                    xerr=all_featimp["sem"],
                                    ecolor='k')
plt.xlabel("Feature importance")
plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate("Average MSE: {0:.2} ± {1:.2}".format(xgb_acc_distrib.mean()[0], xgb_acc_distrib.sem()[0]), xy=(0.04,0), fontsize=8, color="k")

plt.savefig("./plots/xgb_real_age_feat_imp.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_real_age_feat_imp.png", bbox_inches="tight")




#########################################
# PREDICTING REAL AGE OF AG
#########################################

# Spot-checking

#%%
X = df_real_age_AG
y = df_real_age_AG.index
# cross validation
seed = 4
validation_size = 0.25
num_splits = 20
num_repeats = 10
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=num_splits, random_state=seed)


models = []
models.append(("LR", LinearRegression()))
models.append(("EN", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=num_splits, random_state=seed)))
# models.append(("SGD", SGDRegressor(random_state=seed)))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNNR", KNeighborsRegressor()))
# models.append(("CART", DecisionTreeRegressor(random_state=seed)))
models.append(("RF", RandomForestRegressor(random_state=seed)))
# models.append(("ET", ExtraTreeRegressor(random_state=seed)))
models.append(("XGB", XGBRegressor(random_state=seed)))
# models.append(("kNR", KNeighborsRegressor))
# models.append(("ADA", AdaBoostRegressor))

# generate results for each model in turn
results = []
names = []
scoring = "neg_mean_squared_error"

for name, model in models:
    kfold = KFold(n_splits=num_splits, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2} ± {2:.2}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)


# plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Anopheles gambiae real age")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_real_age_AG.png", bbox_inches="tight")
plt.savefig("./plots/spot_check_real_age_AG.pdf", bbox_inches="tight")


## Elastic Net optimisation

#%%
# load data
X = df_real_age_AG
y = df_real_age_AG.index

X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

# cross validation
seed = 4
validation_size = 0.3
num_splits = 5
num_repeats = 10
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=num_splits, random_state=seed)

# base algorithm settings
regressor = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                         eps=0.001,
                         n_alphas=100,
                         alphas=None,
                         fit_intercept=True,
                         normalize=False,
                         precompute="auto",
                         max_iter=10000,
                         tol=0.0001,
                         cv=num_splits,
                         copy_X=True,
                         verbose=1,
                         n_jobs=-1,
                         positive=False,
                         random_state=seed,
                         selection="cyclic")


# repeated random stratified splitting of dataset
rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rkf_coef = pd.DataFrame(columns=["wavelength", "coef"]).set_index("wavelength")
rkf_coef.loc["r2"] = ""
rkf_intercept = pd.DataFrame(columns=["wavelength", "intercept"]).set_index("wavelength")
rkf_intercept.loc["r2"] = ""

for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #fit model
    regressor.fit(X_train, y_train)
    # test model
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    coef_table = pd.DataFrame({"coef":regressor.coef_, "wavelength":X.columns}).set_index("wavelength")
    coef_table.loc["r2"] = r2
    coef_table.loc["mse"] = mse
    intercept_table = pd.DataFrame({"intercept":regressor.intercept_, "wavelength":X.columns}).set_index("wavelength")
    intercept_table.loc["r2"] = r2
    intercept_table.loc["mse"] = mse
    # combine outputs
    rkf_coef = pd.merge(rkf_coef, coef_table, left_index=True, right_index=True, how='outer')
    rkf_intercept = pd.merge(rkf_intercept, intercept_table, left_index=True, right_index=True, how='outer')

# Results
rkf_coef.dropna(axis=1, inplace=True)
rkf_coef["coef mean"] = rkf_coef.mean(axis=1)
rkf_coef["coef sem"] = rkf_coef.sem(axis=1)
rkf_coef.to_csv("enet_AG_age_repeatedCV_coef.csv")
rkf_coef = pd.read_csv("enet_AG_age_repeatedCV_coef.csv", index_col="wavelength")
rkf_intercept.dropna(axis=1, inplace=True)
rkf_intercept["intercept mean"] = rkf_intercept.mean(axis=1)
rkf_intercept["intercept sem"] = rkf_intercept.sem(axis=1)
rkf_intercept["intercept mean"] = rkf_intercept.mean(axis=1)
rkf_intercept["intercept sem"] = rkf_intercept.sem(axis=1)
rkf_intercept.to_csv("enet_AG_age_repeatedCV_intercept.csv")
rkf_intercept = pd.read_csv("enet_AG_age_repeatedCV_intercept.csv", index_col="wavelength")

# # Accuracy distribution
# xgb_acc_distrib = rkf_coef["Accuracy"]
# xgb_acc_distrib.columns=["Accuracy"]
# xgb_acc_distrib.to_csv("xgb_AG_age_acc_distrib.csv", header=True, index=False)rkf_coef.loc[ "r2" ]
print("Intercept: {0:.3} ± {1:.3}".format(rkf_intercept.loc[rkf_coef.index[1]]["intercept mean"], rkf_intercept.loc[rkf_coef.index[1]]["intercept sem"]))
print("Average mse: {0:.2f} ± {1:.2f}\n".format(rkf_coef.loc["mse", ["coef mean"]][0], rkf_coef.loc["mse", ["coef sem"]][0]))
rkf_coef.sort_values(by="coef mean", inplace=True)
print(rkf_coef[~rkf_coef.index.isin(["r2", "mse"])][["coef mean", "coef sem"]].head())
print(rkf_coef[~rkf_coef.index.isin(["r2", "mse"])][["coef mean", "coef sem"]].tail())



Intercept: 9.92 ± 0.195
Average mse: 14.11 ± 0.18

            coef mean  coef sem
wavelength
1746        -4.329662  0.047323
1075        -2.467016  0.031879
1901        -2.181250  0.015545
1636        -2.032161  0.047567
3856        -0.279591  0.017224
            coef mean  coef sem
wavelength
1307         0.001654  0.001621
3400         0.009843  0.003890
880          0.070270  0.024037
1457         4.183307  0.052724
2854         4.899504  0.058470

## plot coeficients

#%%
rkf_coef = pd.read_csv("enet_AG_age_repeatedCV_coef.csv", index_col="wavelength")
rkf_coef.sort_values(by="coef mean", ascending=False, inplace=True)
coef_plot_data = rkf_coef[~rkf_coef.index.isin(["r2", "mse", "AG_age", "log_nw"])].copy()
coef_plot_data = coef_plot_data[coef_plot_data["coef mean"].abs() > 0.001].drop(["coef mean", "coef sem"], axis=1).T


sns.set(context="paper",
        style="white",
        font_scale=1.6,
        rc={"font.family": "Helvetica"})

f = plt.figure(figsize=(4,8))
f = sns.barplot(data=coef_plot_data, orient="h", palette="coolwarm")

plt.xticks(np.arange(np.round(min(rkf_coef["coef mean"][~rkf_coef.index.isin(["r2", "mse", "AG_age", "log_nw"])]), decimals=3), np.round(max(rkf_coef["coef mean"][~rkf_coef.index.isin(["r2", "mse", "AG_age", "log_nw"])]), decimals=3), 0.02), fontsize=10)

f.set_xlabel("Elastic Net average coefficient\n$MSE: {0:.3f} ± {1:.3f}$".format(rkf_coef.loc["mse", ["coef mean"]][0], rkf_coef.loc["mse", ["coef sem"]][0]), weight="bold")

f.set_ylabel("")

plt.savefig("./plots/as_enet_coef_immune_nw.png", bbox_inches="tight")
plt.savefig("./plots/as_enet_coef_immune_nw.pdf", bbox_inches="tight")




#########################################
# PREDICTING REAL AGE OF AR
#########################################

# Spot-checking

#%%
X = df_real_age_AR
y = df_real_age_AR.index

# cross validation
seed = 4
validation_size = 0.25
num_splits = 10
num_repeats = 10
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=num_splits, random_state=seed)

models = []
models.append(("LR", LinearRegression()))
models.append(("EN", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=num_splits, random_state=seed)))
# models.append(("SGD", SGDRegressor(random_state=seed)))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNNR", KNeighborsRegressor()))
models.append(("CART", DecisionTreeRegressor(random_state=seed)))
models.append(("RF", RandomForestRegressor(random_state=seed)))
models.append(("ET", ExtraTreeRegressor(random_state=seed)))
models.append(("XGB", XGBRegressor(random_state=seed)))
# models.append(("kNR", KNeighborsRegressor))
# models.append(("ADA", AdaBoostRegressor))

# generate results for each model in turn
results = []
names = []
scoring = "neg_mean_squared_error"

for name, model in models:
    kfold = KFold(n_splits=num_splits, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2} ± {2:.2}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)


# plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Anopheles arabiensis real age")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_real_age_AR.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_real_age_AR.png", bbox_inches="tight")


## search best parameters
#%%
# load data
X = df_real_age_AR.values
y = df_real_age_AR.index

# cross validation
seed = 4
validation_size = 0.25
num_splits = 10
num_repeats = 10
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=num_splits, random_state=seed)

# base algorithm settings
regressor = XGBRegressor(objective="reg:linear", nthread=1)

# define hyperparameter space to test
## for troubleshooting:
# max_depth = [2, 4]
# min_child_weight = [3, 7]
# learning_rate = [0.01, 0.1]
# n_estimators = [5, 10, 20]

# the real deal:
max_depth = [2, 4, 6, 8] #4?
min_child_weight = [1, 3, 5, 7]
n_estimators = [5, 10, 15, 20, 25, 30]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2] #0.1?
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1] #0.2?


parameters = {"max_depth": max_depth,
              "min_child_weight": min_child_weight,
              "learning_rate": learning_rate,
              "n_estimators": n_estimators,
              "colsample_bytree":colsample_bytree}

# repeated random stratified splitting of dataset
rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rkf_results = pd.DataFrame() # model parameters and global accuracy score
rkf_per_class_results = [] # per class accuracy scores

for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # GRID SEARCH
    gsCV = GridSearchCV(estimator=regressor, param_grid=parameters, scoring=scoring, cv=kfold, n_jobs=-1)
    CV_result = gsCV.fit(X_train, y_train)
    best_model = XGBRegressor(objective="reg:linear", nthread=1, **CV_result.best_params_)

    #fit model
    best_model.fit(X_train, y_train)

    #test model
    y_pred = best_model.predict(X_test)

    local_feat_impces = pd.DataFrame(best_model.feature_importances_, index=df_real_age.columns).sort_values(by=0, ascending=False)

    local_rkf_results = pd.DataFrame([("Accuracy",mean_squared_error(y_test, y_pred)), ("params",str(CV_result.best_params_)), ("seed", best_model.seed), ("TRAIN",str(train_index)), ("TEST",str(test_index)),  ("Feature importances", local_feat_impces.to_dict())]).T

    # combine outputs
    local_rkf_results.columns=local_rkf_results.iloc[0]
    local_rkf_results = local_rkf_results[1:]
    rkf_results = rkf_results.append(local_rkf_results)


# Results
rkf_results.to_csv("xgb_real_age_AR_repeatedCV_record.csv", index=False)
rkf_results = pd.read_csv("xgb_real_age_AR_repeatedCV_record.csv")


# Accuracy distribution
xgb_acc_distrib = rkf_results["Accuracy"]
xgb_acc_distrib.columns=["Accuracy"]
xgb_acc_distrib.to_csv("xgb_real_age_AR_acc_distrib.csv", header=True, index=False)



# Feature Importances

#%%
rskf_results = pd.read_csv("xgb_real_age_AR_repeatedCV_record.csv")
xgb_acc_distrib = pd.read_csv("xgb_real_age_AR_acc_distrib.csv")

# All feat imp
all_featimp = pd.DataFrame(ast.literal_eval(rskf_results["Feature importances"][0]))

for featimp in rskf_results["Feature importances"][1:]:
    featimp = pd.DataFrame(ast.literal_eval(featimp))
    all_featimp = all_featimp.merge(featimp, left_index=True, right_index=True)

all_featimp["mean"] = all_featimp.mean(axis=1)
all_featimp["sem"] = all_featimp.sem(axis=1)
all_featimp.sort_values(by="mean", inplace=True)

featimp_global_mean = all_featimp["mean"].mean()
featimp_global_sem = all_featimp["mean"].sem()


fig = all_featimp["mean"][-13:].plot(figsize=(2.2, 4),
                                    kind="barh",
                                    legend=False,
                                    xerr=all_featimp["sem"],
                                    ecolor='k')
plt.xlabel("Feature importance")
plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate("Average MSE: {0:.2} ± {1:.2}".format(xgb_acc_distrib.mean()[0], xgb_acc_distrib.sem()[0]), xy=(0.04,0), fontsize=8, color="k")

plt.savefig("./plots/xgb_real_age_AR_feat_imp.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_real_age_AR_feat_imp.png", bbox_inches="tight")


