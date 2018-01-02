# %% import modules
import os
import warnings

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
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import maxabs_scale

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from xgboost import XGBClassifier
from xgboost import XGBRegressor

import hdbscan

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
    Normalization can be applied by setting `normalize=True`.
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

#%% import data
df_real_age = pd.read_table("./mosquitoes_spectra (170623).dat", index_col="Real age")
df_real_age = df_real_age.ix[:, 4:-1]
df_real_age.head()

#%% 
df_real_age[df_real_age.columns] = StandardScaler().fit_transform(df_real_age[df_real_age.columns].as_matrix())
df_real_age.head()

# %% Spot check classification models
df = df_real_age.copy()
y = df.index
X = df


# cross-val settings
seed = 4
validation_size = 0.30
num_folds = 10
num_splits = 10

# pick the models to test
models = []
# models.append(("LR", LinearRegression()))
models.append(("EN", ElasticNetCV()))
# models.append(("SGD", SGDRegressor()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNNR", KNeighborsRegressor()))
# models.append(("CART", DecisionTreeRegressor()))
models.append(("RF", RandomForestRegressor()))
# models.append(("ET", ExtraTreeRegressor()))
models.append(("XGB", XGBRegressor()))
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


#%% plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Worm burden")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_real_age.pdf", bbox_inches="tight")


#%% ElasticNet

# load data
df = df_real_age.copy()
df[df.columns] = StandardScaler().fit_transform(df[df.columns].as_matrix())

y = df.index
X = df

X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

# cross validation
seed = 4
validation_size = 0.30
num_splits = 10
num_repeats = 4
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
rkf = RepeatedKFold(n_splits=num_splits,
                    n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rkf_coef = pd.DataFrame(
    columns=["wavelength", "coef"]).set_index("wavelength")
rkf_coef.loc["r2"] = ""
rkf_intercept = pd.DataFrame(
    columns=["wavelength", "intercept"]).set_index("wavelength")
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
    coef_table = pd.DataFrame(
        {"coef": regressor.coef_, "wavelength": X.columns}).set_index("wavelength")
    coef_table.loc["r2"] = r2
    coef_table.loc["mse"] = mse
    intercept_table = pd.DataFrame(
        {"intercept": regressor.intercept_, "wavelength": X.columns}).set_index("wavelength")
    intercept_table.loc["r2"] = r2
    intercept_table.loc["mse"] = mse
    # combine outputs
    rkf_coef = pd.merge(rkf_coef, coef_table, left_index=True,
                        right_index=True, how='outer')
    rkf_intercept = pd.merge(
        rkf_intercept, intercept_table, left_index=True, right_index=True, how='outer')

# Results
rkf_coef.dropna(axis=1, inplace=True)
rkf_coef["coef mean"] = rkf_coef.mean(axis=1)
rkf_coef["coef sem"] = rkf_coef.sem(axis=1)
rkf_coef.to_csv("enet_real_age_repeatedCV_coef.csv")
rkf_coef = pd.read_csv("enet_real_age_repeatedCV_coef.csv", index_col="wavelength")
rkf_intercept.dropna(axis=1, inplace=True)
rkf_intercept["intercept mean"] = rkf_intercept.mean(axis=1)
rkf_intercept["intercept sem"] = rkf_intercept.sem(axis=1)
rkf_intercept["intercept mean"] = rkf_intercept.mean(axis=1)
rkf_intercept["intercept sem"] = rkf_intercept.sem(axis=1)
rkf_intercept.to_csv("enet_real_age_repeatedCV_intercept.csv")
rkf_intercept = pd.read_csv(
    "enet_real_age_repeatedCV_intercept.csv", index_col="wavelength")

# # Accuracy distribution
# xgb_acc_distrib = rkf_coef["Accuracy"]
# xgb_acc_distrib.columns=["Accuracy"]
# xgb_acc_distrib.to_csv("xgb_real_age_acc_distrib.csv", header=True, index=False)rkf_coef.loc[ "r2" ]
print("Intercept: {0:.3} ± {1:.3}".format(
    rkf_intercept.loc[rkf_coef.index[1]]["intercept mean"], rkf_intercept.loc[rkf_coef.index[1]]["intercept sem"]))
print("Average mse: {0:.2f} ± {1:.2f}\n".format(
    rkf_coef.loc["mse", ["coef mean"]][0], rkf_coef.loc["mse", ["coef sem"]][0]))
rkf_coef.sort_values(by="coef mean", inplace=True)
print(rkf_coef[~rkf_coef.index.isin(["r2", "mse"])]
      [["coef mean", "coef sem"]].head())
print(rkf_coef[~rkf_coef.index.isin(["r2", "mse"])]
      [["coef mean", "coef sem"]].tail())


#%% plot coeficients


rkf_coef = pd.read_csv("enet_real_age_repeatedCV_coef.csv", index_col="wavelength")
rkf_coef.sort_values(by="coef mean", ascending=False, inplace=True)
coef_plot_data = rkf_coef[~rkf_coef.index.isin(
    ["r2", "mse", "nw_counts", "log_real_age"])].copy()
coef_plot_data = coef_plot_data[coef_plot_data["coef mean"].abs() > 0.001].drop(
    ["coef mean", "coef sem"], axis=1).T


sns.set(context="paper",
        style="white",
        font_scale=1.6,
        rc={"font.family": "Helvetica"})

f = plt.figure(figsize=(4, 8))
f = sns.barplot(data=coef_plot_data, orient="h", palette="coolwarm")

plt.xticks(np.arange(np.round(min(rkf_coef["coef mean"][~rkf_coef.index.isin(["r2", "mse", "nw_counts", "log_real_age"])]), decimals=3), np.round(
    max(rkf_coef["coef mean"][~rkf_coef.index.isin(["r2", "mse", "nw_counts", "log_real_age"])]), decimals=3), 0.02), fontsize=10)

f.set_xlabel("Elastic Net coefficient\n$MSE: {0:.3f} ± {1:.3f}$".format(
    rkf_coef.loc["mse", ["coef mean"]][0], rkf_coef.loc["mse", ["coef sem"]][0]), weight="bold")

f.set_ylabel("")

plt.savefig("./plots/as_enet_coef_real_age.pdf", bbox_inches="tight")



# %% XGBRegressor

# Load data
df = df_real_age.copy()

y = df.index
X = df

# cross validation
seed = 4
validation_size = 0.30
num_splits = 10
num_repeats = 2
scoring = "neg_mean_squared_error"
kfold = KFold(n_splits=num_splits, random_state=seed)

# base algorithm settings
regressor = XGBRegressor(nthread=1)

# define hyperparameter space to test
## for troubleshooting:
# max_depth = [2, 4]
# min_child_weight = [3, 7]
# learning_rate = [0.01, 0.1]
# n_estimators = [5, 10, 20]

# the real deal:
max_depth = [2, 4, 6, 8]  # 4?
min_child_weight = [1, 3, 5, 7]
n_estimators = [5, 10, 15, 20, 25, 30]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2]  # 0.1?
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]  # 0.2?


parameters = {"max_depth": max_depth,
              "min_child_weight": min_child_weight,
              "learning_rate": learning_rate,
              "n_estimators": n_estimators,
              "colsample_bytree": colsample_bytree}

# repeated random stratified splitting of dataset
rkf = RepeatedKFold(n_splits=num_splits,
                    n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rkf_results = pd.DataFrame()  # model parameters and global accuracy score
rkf_per_class_results = []  # per class accuracy scores

for train_index, test_index in rkf.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # GRID SEARCH
    gsCV = GridSearchCV(estimator=regressor, param_grid=parameters,
                        scoring=scoring, cv=kfold, n_jobs=-1)
    CV_result = gsCV.fit(X_train, y_train)
    best_model = XGBRegressor(nthread=1, **CV_result.best_params_)

    #fit model
    best_model.fit(X_train, y_train)

    #test model
    y_pred = best_model.predict(X_test)

    local_feat_impces = pd.DataFrame(
        best_model.feature_importances_, index=df_real_age.columns).sort_values(by=0, ascending=False)

    local_rkf_results = pd.DataFrame([("Accuracy", mean_squared_error(y_test, y_pred)), ("params", str(CV_result.best_params_)), (
        "seed", best_model.seed), ("TRAIN", str(train_index)), ("TEST", str(test_index)),  ("Feature importances", local_feat_impces.to_dict())]).T

    # combine outputs
    local_rkf_results.columns = local_rkf_results.iloc[0]
    local_rkf_results = local_rkf_results[1:]
    rkf_results = rkf_results.append(local_rkf_results)


# Results
rkf_results.to_csv("xgb_real_age_repeatedCV_record.csv", index=False)
rkf_results = pd.read_csv("xgb_real_age_repeatedCV_record.csv")


# Accuracy distribution
xgb_acc_distrib = rkf_results["Accuracy"]
xgb_acc_distrib.columns = ["Accuracy"]
xgb_acc_distrib.to_csv("xgb_real_age_acc_distrib.csv",
                       header=True, index=False)

# %% Plots 

# Feature Importances

rskf_results = pd.read_csv("xgb_real_age_repeatedCV_record.csv")
xgb_acc_distrib = pd.read_csv("xgb_real_age_acc_distrib.csv")

# All feat imp
all_featimp = pd.DataFrame(ast.literal_eval(
    rskf_results["Feature importances"][0]))

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
plt.axvspan(xmin=0, xmax=featimp_global_mean + 3 *
            featimp_global_sem, facecolor='r', alpha=0.3)
plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate("Average MSE: {0:.2} ± {1:.2}".format(xgb_acc_distrib.mean()[
             0], xgb_acc_distrib.sem()[0]), xy=(0.04, 0), fontsize=8, color="k")

plt.savefig("./plots/xgb_real_age_feat_imp.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_real_age_feat_imp.png", bbox_inches="tight")

