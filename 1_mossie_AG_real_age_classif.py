# %% import modules
import os
import warnings
from time import time
from tqdm import tqdm
from collections import Counter
import ast
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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_fscore_support, mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import maxabs_scale

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from xgboost import XGBClassifier

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

HOME = os.path.expanduser("~")  # just in case we need this later

warnings.filterwarnings(
    "ignore", message="Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable oob estimates")

# Set random seed to insure reproducibility
seed = 4

## FUNCTIONS


def plot_confusion_matrix(cm, classes,
                          normalise=True,
                          text=False,
                          title='Confusion matrix',
                          xrotation=0,
                          yrotation=0,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """

    if normalise:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "{0} (normalised)".format(title)
            # print("Normalized confusion matrix")
        # else:
            # print('Confusion matrix')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

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


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    # From https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(
            dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(
        dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(
        dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()
                         ).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


#%% import data

# import full dataset
df_full = pd.read_table("mosquitoes_data (2018).dat")

df_full.head()


df_real_age = pd.read_table( "./mosquitoes_data (2018).dat", index_col="Age")
df_real_age_elim = df_real_age.loc[[1, 3, 5, 7, 9, 11, 13, 15, 17], :]

df_ag_real_age = df_real_age_elim[df_real_age_elim["Species"] == "AG"]
df_ag_real_age = df_ag_real_age.iloc[:, 3:-1]
df_ag_real_age.head()

df_ag_real_age[df_ag_real_age.columns] = StandardScaler().fit_transform(
    df_ag_real_age[df_ag_real_age.columns].as_matrix())
df_ag_real_age.head()

df_ag_real_age.index.unique()

# feed
df_all_food = df_full.copy()
df_all_food.index = df_all_food["Status"]
df_all_food = df_all_food.iloc[:, 4:]

df_ag_food = df_full.copy()
df_ag_food.index = df_ag_food["Age"]
df_ag_food_elim = df_ag_food.loc[[1, 3, 5, 7, 9, 11, 13, 15, 17], :]

df_ag_SF = df_ag_food_elim[df_ag_food_elim["Status"] == "SF"].iloc[:, 4:]
df_ag_SF = df_ag_food_elim[df_ag_food_elim["Status"] == "SF"].iloc[:, 4:]
df_ag_GR = df_ag_food_elim[df_ag_food_elim["Status"] == "GR"].iloc[:, 4:]

# transform species data
for df in [df_ag_SF, df_ag_SF, df_ag_GR]:
    df[df.columns] = StandardScaler().fit_transform(df[df.columns].as_matrix())


# %% Spot check classification models
df = df_ag_real_age.copy()
y = df.index
X = df.values

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=34)
X_resampled, y_resampled = rus.fit_sample(X, y)

# cross-val settings
seed = 4
validation_size = 0.30
num_splits = 10

# pick the models to test
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
    sss.split(X, y)
    cv_results = cross_val_score(
        model, X, y, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)

#%% plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results, whis=10)
sns.despine(offset=10, trim=True)
plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_ag_real_age.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_ag_real_age.png", bbox_inches="tight")


#%% Logistic Regression

# load data
df = df_ag_real_age.copy()

y = df.index
X = df.values

# cross validation
validation_size = 0.3
num_splits = 5
num_repeats = 10
num_rounds = 3
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
rskf = RepeatedStratifiedKFold(
    n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame()  # model parameters and global accuracy score
rskf_per_class_results = []  # per class accuracy scores
rkf_scores = pd.DataFrame(
    columns=["age", "scores mean", "scores sem"]).set_index("age")

start = time()

for round in range(num_rounds):
    seed = np.random.randint(0, 81470108)

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit model
        classifier.fit(X_train, y_train)

        #test model
        y_pred = classifier.predict(np.delete(X, train_index, axis=0))
        y_predproba = classifier.predict_proba(
            np.delete(X, train_index, axis=0))
        y_test = np.delete(y, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)
        local_scores = pd.DataFrame.from_records([classifier.scores_])

        scores_table = pd.DataFrame({"scores": pd.Series(
            classifier.scores_), "age": df.index.unique()}).set_index("age")

        # combine score outputs
        rkf_scores = pd.merge(rkf_scores, scores_table,
                              left_index=True, right_index=True, how='outer')

        local_rskf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)), ("TRAIN", str(train_index)), ("TEST", str(
            test_index)), ("Pred_probas", y_predproba), ("CM", local_cm), ("Classification report", local_report), ("Scores", local_scores)]).T

        local_rskf_results.columns = local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm) / local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv("./results/lr_ag_age_repeatedCV_record.csv", index=False)

# rename columns to avoid duplicates
nameslist = list(range(1, rkf_scores.shape[1] - 1))
rkf_scores.columns = rkf_scores.columns[:2].tolist() + nameslist

# calculate mean and sem of scores
rownum = 0
for rowid in rkf_scores.index:
    rkf_scores.at[rowid, "scores mean"] = np.mean(
        np.mean(rkf_scores.iloc[rownum, 3:]))
    rkf_scores.at[rowid, "scores sem"] = np.mean(
        stats.sem(np.mean(rkf_scores.iloc[rownum, 3:])))
    rownum += 1


rkf_scores.dropna(axis=1).to_csv(
    "./results/lr_ag_age_rkf_scores.csv", index=False)

rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record.csv")
rkf_scores = pd.read_csv("./results/lr_ag_age_rkf_scores.csv")
# rkf_scores.index = y.unique().zfill(1)


# Accuracy distribution
lr_ag_age_acc_distrib = rskf_results["Accuracy"]
lr_ag_age_acc_distrib.columns = ["Accuracy"]
lr_ag_age_acc_distrib.to_csv(
    "./results/lr_ag_age_acc_distrib.csv", header=True, index=False)
lr_ag_age_acc_distrib = pd.read_csv("./results/lr_ag_age_acc_distrib.csv")
lr_ag_age_acc_distrib = np.round(100 * lr_ag_age_acc_distrib)

plt.figure(figsize=(2.25, 3))
sns.distplot(lr_ag_age_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/lr_ag_age_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_acc_distrib.png", bbox_inches="tight")

# Per class accuracy
class_names = y.sort_values().unique()
lr_ag_age_per_class_acc_distrib = pd.DataFrame(
    rskf_per_class_results, columns=class_names)
lr_ag_age_per_class_acc_distrib.dropna().to_csv(
    "./results/lr_ag_age_per_class_acc_distrib.csv")
lr_ag_age_per_class_acc_distrib = pd.read_csv(
    "./results/lr_ag_age_per_class_acc_distrib.csv", index_col=0)
lr_ag_age_per_class_acc_distrib = np.round(
    100 * lr_ag_age_per_class_acc_distrib)
lr_ag_age_per_class_acc_distrib_describe = lr_ag_age_per_class_acc_distrib.describe()
lr_ag_age_per_class_acc_distrib_describe.to_csv(
    "./results/lr_ag_age_per_class_acc_distrib.csv")

lr_ag_age_per_class_acc_distrib = pd.melt(
    lr_ag_age_per_class_acc_distrib, var_name="AR age")
lr_ag_age_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="AR age", y="value", cut=0,
               order=list(lr_ag_age_per_class_acc_distrib["AR age"].unique()),
               data=lr_ag_age_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xlabel("Anopheles gambiae age")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib.png", bbox_inches="tight")

#%% Confusion matrix
rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record.csv")

best_cm = rskf_results.sort_values(
    by="Accuracy", ascending=False).iloc[0, 4].replace('[ ', '[').split()
best_cm = np.array(ast.literal_eval(",".join(best_cm)))

class_names = y.sort_values().unique()

plt.figure(figsize=(4, 4))
plot_confusion_matrix(best_cm, classes=class_names)
plt.savefig("./plots/lr_ag_age_cm.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_cm.png", bbox_inches="tight")

# %% XGBRegressor

# Load data
df = df_ag_real_age.copy()

y = df.index
X = df
Counter(y)

# cross validation
seed = 4
validation_size = 0.30
num_splits = 10
num_repeats = 10
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
                        scoring=scoring, cv=kfold, n_jobs=-1, verbose=1)
    CV_result = gsCV.fit(X_train, y_train)
    best_model = XGBRegressor(nthread=1, **CV_result.best_params_)

    #fit model
    best_model.fit(X_train, y_train)

    #test model
    y_pred = best_model.predict(X_test)

    local_feat_impces = pd.DataFrame(
        best_model.feature_importances_, index=df_ag_real_age.columns).sort_values(by=0, ascending=False)

    local_rkf_results = pd.DataFrame([("Accuracy", mean_squared_error(y_test, y_pred)), ("params", str(CV_result.best_params_)), (
        "seed", best_model.seed), ("TRAIN", str(train_index)), ("TEST", str(test_index)),  ("Feature importances", local_feat_impces.to_dict())]).T

    # combine outputs
    local_rkf_results.columns = local_rkf_results.iloc[0]
    local_rkf_results = local_rkf_results[1:]
    rkf_results = rkf_results.append(local_rkf_results)


# Results
rkf_results.to_csv("xgb_ag_real_age_repeatedCV_record.csv", index=False)
rkf_results = pd.read_csv("xgb_ag_real_age_repeatedCV_record.csv")


# Accuracy distribution
xgb_acc_distrib = rkf_results["Accuracy"]
xgb_acc_distrib.columns = ["Accuracy"]
xgb_acc_distrib.to_csv("xgb_ag_real_age_acc_distrib.csv",
                       header=True, index=False)

# %% Plot Feature Importances

rskf_results = pd.read_csv("xgb_ag_real_age_repeatedCV_record.csv")
xgb_acc_distrib = pd.read_csv("xgb_ag_real_age_acc_distrib.csv")

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
plt.annotate("Average MSE:\n{0:.2} ± {1:.2}".format(xgb_acc_distrib.mean()[
             0], xgb_acc_distrib.sem()[0]), xy=(0.07, 0), fontsize=8, color="k")

plt.savefig("./plots/xgb_ag_real_age_feat_imp.pdf", bbox_inches="tight")
plt.savefig("./plots/xgb_ag_real_age_feat_imp.png", bbox_inches="tight")


###### Blood-fed ########
# %% Spot check classification models
df = df_ag_BF.copy()
y = df.index
X = df.values

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=34)
X_resampled, y_resampled = rus.fit_sample(X, y)

# cross-val settings
seed = 4
validation_size = 0.30
num_splits = 10

# pick the models to test
models = []
models.append(("KNN", KNeighborsClassifier()))
# models.append(("SGD", SGDClassifier()))
# models.append(("LDA", LinearDiscriminantAnalysis(random_state=seed)))
models.append(("LR", LogisticRegressionCV()))
# models.append(("CART", DecisionTreeClassifier()))
models.append(("SVM", SVC()))
# models.append(("NB", GaussianNB()))
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
    sss.split(X, y)
    cv_results = cross_val_score(
        model, X, y, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)

#%% plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_ag_real_age_BF.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_ag_real_age_BF.png", bbox_inches="tight")


#%% Logistic Regression

# load data
df = df_ag_BF.copy()

y = df.index
X = df.values

# cross validation
validation_size = 0.3
num_splits = 10
num_repeats = 10
num_rounds = 5
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
rskf = RepeatedStratifiedKFold(
    n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame()  # model parameters and global accuracy score
rskf_per_class_results = []  # per class accuracy scores
rkf_scores = pd.DataFrame(
    columns=["age", "scores mean", "scores sem"]).set_index("age")

start = time()

for round in range(num_rounds):
    seed = np.random.randint(0, 81470108)

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit model
        classifier.fit(X_train, y_train)

        #test model
        y_pred = classifier.predict(np.delete(X, train_index, axis=0))
        y_predproba = classifier.predict_proba(
            np.delete(X, train_index, axis=0))
        y_test = np.delete(y, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)
        local_scores = pd.DataFrame.from_records([classifier.scores_])

        scores_table = pd.DataFrame({"scores": pd.Series(
            classifier.scores_), "age": df.index.unique()}).set_index("age")

        # combine score outputs
        rkf_scores = pd.merge(rkf_scores, scores_table,
                              left_index=True, right_index=True, how='outer')

        local_rskf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)), ("TRAIN", str(train_index)), ("TEST", str(
            test_index)), ("Pred_probas", y_predproba), ("CM", local_cm), ("Classification report", local_report), ("Scores", local_scores)]).T

        local_rskf_results.columns = local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm) / local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv(
    "./results/lr_ag_age_repeatedCV_record_BF.csv", index=False)

# rename columns to avoid duplicates
nameslist = list(range(1, rkf_scores.shape[1] - 1))
rkf_scores.columns = rkf_scores.columns[:2].tolist() + nameslist

# calculate mean and sem of scores
rownum = 0
for rowid in rkf_scores.index:
    rkf_scores.at[rowid, "scores mean"] = np.mean(
        np.mean(rkf_scores.iloc[rownum, 3:]))
    rkf_scores.at[rowid, "scores sem"] = np.mean(
        stats.sem(np.mean(rkf_scores.iloc[rownum, 3:])))
    rownum += 1


rkf_scores.dropna(axis=1).to_csv(
    "./results/lr_ag_age_rkf_scores_BF.csv", index=False)

rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record_BF.csv")
rkf_scores = pd.read_csv("./results/lr_ag_age_rkf_scores_BF.csv")
# rkf_scores.index = y.unique().zfill(1)


# Accuracy distribution
lr_ag_age_acc_distrib = rskf_results["Accuracy"]
lr_ag_age_acc_distrib.columns = ["Accuracy"]
lr_ag_age_acc_distrib.to_csv(
    "./results/lr_ag_age_acc_distrib_BF.csv", header=True, index=False)
lr_ag_age_acc_distrib = pd.read_csv("./results/lr_ag_age_acc_distrib_BF.csv")
lr_ag_age_acc_distrib = np.round(100 * lr_ag_age_acc_distrib)

plt.figure(figsize=(2.25, 3))
sns.distplot(lr_ag_age_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/lr_ag_age_acc_distrib_BF.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_acc_distrib_BF.png", bbox_inches="tight")

# Per class accuracy
class_names = y.sort_values().unique()
lr_ag_age_per_class_acc_distrib = pd.DataFrame(
    rskf_per_class_results, columns=class_names)
lr_ag_age_per_class_acc_distrib.dropna().to_csv(
    "./results/lr_ag_age_per_class_acc_distrib_BF.csv")
lr_ag_age_per_class_acc_distrib = pd.read_csv(
    "./results/lr_ag_age_per_class_acc_distrib_BF.csv", index_col=0)
lr_ag_age_per_class_acc_distrib = np.round(
    100 * lr_ag_age_per_class_acc_distrib)
lr_ag_age_per_class_acc_distrib_describe = lr_ag_age_per_class_acc_distrib.describe()
lr_ag_age_per_class_acc_distrib_describe.to_csv(
    "./results/lr_ag_age_per_class_acc_distrib_BF.csv")

lr_ag_age_per_class_acc_distrib = pd.melt(
    lr_ag_age_per_class_acc_distrib, var_name="AR age")
lr_ag_age_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="AR age", y="value", cut=0,
               order=list(lr_ag_age_per_class_acc_distrib["AR age"].unique()),
               data=lr_ag_age_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xlabel("Anopheles arabiensis age")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib_BF.pdf",
            bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib_BF.png",
            bbox_inches="tight")

#%% Confusion matrix
rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record_BF.csv")

best_cm = rskf_results.sort_values(
    by="Accuracy", ascending=False).iloc[0, 4].replace('[ ', '[').split()
best_cm = np.array(ast.literal_eval(",".join(best_cm)))

class_names = y.sort_values().unique()

plt.figure(figsize=(4, 4))
plot_confusion_matrix(best_cm, classes=class_names)
plt.savefig("./plots/lr_ag_age_cm_BF.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_cm_BF.png", bbox_inches="tight")


###### Sugar-fed ########
# %% Spot check classification models
df = df_ag_SF.copy()
y = df.index
X = df.values

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=34)
X_resampled, y_resampled = rus.fit_sample(X, y)

# cross-val settings
seed = 4
validation_size = 0.30
num_splits = 10

# pick the models to test
models = []
models.append(("KNN", KNeighborsClassifier()))
# models.append(("SGD", SGDClassifier()))
# models.append(("LDA", LinearDiscriminantAnalysis(random_state=seed)))
models.append(("LR", LogisticRegressionCV()))
# models.append(("CART", DecisionTreeClassifier()))
models.append(("SVM", SVC()))
# models.append(("NB", GaussianNB()))
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
    sss.split(X, y)
    cv_results = cross_val_score(
        model, X, y, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)

#%% plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_ag_real_age_SF.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_ag_real_age_SF.png", bbox_inches="tight")


#%% Logistic Regression

# load data
df = df_ag_SF.copy()

y = df.index
X = df.values

# cross validation
validation_size = 0.3
num_splits = 10
num_repeats = 10
num_rounds = 5
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
rskf = RepeatedStratifiedKFold(
    n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame()  # model parameters and global accuracy score
rskf_per_class_results = []  # per class accuracy scores
rkf_scores = pd.DataFrame(
    columns=["age", "scores mean", "scores sem"]).set_index("age")

start = time()

for round in range(num_rounds):
    seed = np.random.randint(0, 81470108)

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit model
        classifier.fit(X_train, y_train)

        #test model
        y_pred = classifier.predict(np.delete(X, train_index, axis=0))
        y_predproba = classifier.predict_proba(
            np.delete(X, train_index, axis=0))
        y_test = np.delete(y, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)
        local_scores = pd.DataFrame.from_records([classifier.scores_])

        scores_table = pd.DataFrame({"scores": pd.Series(
            classifier.scores_), "age": df.index.unique()}).set_index("age")

        # combine score outputs
        rkf_scores = pd.merge(rkf_scores, scores_table,
                              left_index=True, right_index=True, how='outer')

        local_rskf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)), ("TRAIN", str(train_index)), ("TEST", str(
            test_index)), ("Pred_probas", y_predproba), ("CM", local_cm), ("Classification report", local_report), ("Scores", local_scores)]).T

        local_rskf_results.columns = local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm) / local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv(
    "./results/lr_ag_age_repeatedCV_record_SF.csv", index=False)

# rename columns to avoid duplicates
nameslist = list(range(1, rkf_scores.shape[1] - 1))
rkf_scores.columns = rkf_scores.columns[:2].tolist() + nameslist

# calculate mean and sem of scores
rownum = 0
for rowid in rkf_scores.index:
    rkf_scores.at[rowid, "scores mean"] = np.mean(
        np.mean(rkf_scores.iloc[rownum, 3:]))
    rkf_scores.at[rowid, "scores sem"] = np.mean(
        stats.sem(np.mean(rkf_scores.iloc[rownum, 3:])))
    rownum += 1


rkf_scores.dropna(axis=1).to_csv(
    "./results/lr_ag_age_rkf_scores_SF.csv", index=False)

rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record_SF.csv")
rkf_scores = pd.read_csv("./results/lr_ag_age_rkf_scores_SF.csv")
# rkf_scores.index = y.unique().zfill(1)


# Accuracy distribution
lr_ag_age_acc_distrib = rskf_results["Accuracy"]
lr_ag_age_acc_distrib.columns = ["Accuracy"]
lr_ag_age_acc_distrib.to_csv(
    "./results/lr_ag_age_acc_distrib_SF.csv", header=True, index=False)
lr_ag_age_acc_distrib = pd.read_csv("./results/lr_ag_age_acc_distrib_SF.csv")
lr_ag_age_acc_distrib = np.round(100 * lr_ag_age_acc_distrib)

plt.figure(figsize=(2.25, 3))
sns.distplot(lr_ag_age_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/lr_ag_age_acc_distrib_SF.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_acc_distrib_SF.png", bbox_inches="tight")

# Per class accuracy
class_names = y.sort_values().unique()
lr_ag_age_per_class_acc_distrib = pd.DataFrame(
    rskf_per_class_results, columns=class_names)
lr_ag_age_per_class_acc_distrib.dropna().to_csv(
    "./results/lr_ag_age_per_class_acc_distrib_SF.csv")
lr_ag_age_per_class_acc_distrib = pd.read_csv(
    "./results/lr_ag_age_per_class_acc_distrib_SF.csv", index_col=0)
lr_ag_age_per_class_acc_distrib = np.round(
    100 * lr_ag_age_per_class_acc_distrib)
lr_ag_age_per_class_acc_distrib_describe = lr_ag_age_per_class_acc_distrib.describe()
lr_ag_age_per_class_acc_distrib_describe.to_csv(
    "./results/lr_ag_age_per_class_acc_distrib_SF.csv")

lr_ag_age_per_class_acc_distrib = pd.melt(
    lr_ag_age_per_class_acc_distrib, var_name="AR age")
lr_ag_age_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="AR age", y="value", cut=0,
               order=list(lr_ag_age_per_class_acc_distrib["AR age"].unique()),
               data=lr_ag_age_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xlabel("Anopheles arabiensis age")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib_SF.pdf",
            bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib_SF.png",
            bbox_inches="tight")

#%% Confusion matrix
rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record_SF.csv")

best_cm = rskf_results.sort_values(
    by="Accuracy", ascending=False).iloc[0, 4].replace('[ ', '[').split()
best_cm = np.array(ast.literal_eval(",".join(best_cm)))

class_names = y.sort_values().unique()

plt.figure(figsize=(4, 4))
plot_confusion_matrix(best_cm, classes=class_names)
plt.savefig("./plots/lr_ag_age_cm_SF.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_cm_SF.png", bbox_inches="tight")


###### Gravid ########
# %% Spot check classification models
df = df_ag_GR.copy()
y = df.index
X = df.values

# under-sample over-represented classes
rus = RandomUnderSampler(random_state=34)
X_resampled, y_resampled = rus.fit_sample(X, y)

# cross-val settings
seed = 4
validation_size = 0.30
num_splits = 10

# pick the models to test
models = []
models.append(("KNN", KNeighborsClassifier()))
# models.append(("SGD", SGDClassifier()))
# models.append(("LDA", LinearDiscriminantAnalysis(random_state=seed)))
models.append(("LR", LogisticRegressionCV()))
# models.append(("CART", DecisionTreeClassifier()))
models.append(("SVM", SVC()))
# models.append(("NB", GaussianNB()))
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
    sss.split(X, y)
    cv_results = cross_val_score(
        model, X, y, cv=sss, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
        name, cv_results.mean(), cv_results.std())
    print(msg)

#%% plot
g = plt.figure(figsize=(2.4, 3))
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation=90)
plt.ylabel('Negative mean Squared Error')

plt.savefig("./plots/spot_check_ag_real_age_GR.pdf", bbox_inches="tight")
plt.savefig("./plots/spot_check_ag_real_age_GR.png", bbox_inches="tight")


#%% Logistic Regression

# load data
df = df_ag_GR.copy()

y = df.index
X = df.values

# cross validation
validation_size = 0.3
num_splits = 10
num_repeats = 10
num_rounds = 5
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
rskf = RepeatedStratifiedKFold(
    n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame()  # model parameters and global accuracy score
rskf_per_class_results = []  # per class accuracy scores
rkf_scores = pd.DataFrame(
    columns=["age", "scores mean", "scores sem"]).set_index("age")

start = time()

for round in range(num_rounds):
    seed = np.random.randint(0, 81470108)

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit model
        classifier.fit(X_train, y_train)

        #test model
        y_pred = classifier.predict(np.delete(X, train_index, axis=0))
        y_predproba = classifier.predict_proba(
            np.delete(X, train_index, axis=0))
        y_test = np.delete(y, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)
        local_scores = pd.DataFrame.from_records([classifier.scores_])

        scores_table = pd.DataFrame({"scores": pd.Series(
            classifier.scores_), "age": df.index.unique()}).set_index("age")

        # combine score outputs
        rkf_scores = pd.merge(rkf_scores, scores_table,
                              left_index=True, right_index=True, how='outer')

        local_rskf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)), ("TRAIN", str(train_index)), ("TEST", str(
            test_index)), ("Pred_probas", y_predproba), ("CM", local_cm), ("Classification report", local_report), ("Scores", local_scores)]).T

        local_rskf_results.columns = local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm) / local_support
        rskf_per_class_results.append(local_acc)


elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))

# Results
rskf_results.to_csv(
    "./results/lr_ag_age_repeatedCV_record_GR.csv", index=False)

# rename columns to avoid duplicates
nameslist = list(range(1, rkf_scores.shape[1] - 1))
rkf_scores.columns = rkf_scores.columns[:2].tolist() + nameslist

# calculate mean and sem of scores
rownum = 0
for rowid in rkf_scores.index:
    rkf_scores.at[rowid, "scores mean"] = np.mean(
        np.mean(rkf_scores.iloc[rownum, 3:]))
    rkf_scores.at[rowid, "scores sem"] = np.mean(
        stats.sem(np.mean(rkf_scores.iloc[rownum, 3:])))
    rownum += 1


rkf_scores.dropna(axis=1).to_csv(
    "./results/lr_ag_age_rkf_scores_GR.csv", index=False)

rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record_GR.csv")
rkf_scores = pd.read_csv("./results/lr_ag_age_rkf_scores_GR.csv")
# rkf_scores.index = y.unique().zfill(1)


# Accuracy distribution
lr_ag_age_acc_distrib = rskf_results["Accuracy"]
lr_ag_age_acc_distrib.columns = ["Accuracy"]
lr_ag_age_acc_distrib.to_csv(
    "./results/lr_ag_age_acc_distrib_GR.csv", header=True, index=False)
lr_ag_age_acc_distrib = pd.read_csv("./results/lr_ag_age_acc_distrib_GR.csv")
lr_ag_age_acc_distrib = np.round(100 * lr_ag_age_acc_distrib)

plt.figure(figsize=(2.25, 3))
sns.distplot(lr_ag_age_acc_distrib, kde=False, bins=12)
plt.savefig("./plots/lr_ag_age_acc_distrib_GR.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_acc_distrib_GR.png", bbox_inches="tight")

# Per class accuracy
class_names = y.sort_values().unique()
lr_ag_age_per_class_acc_distrib = pd.DataFrame(
    rskf_per_class_results, columns=class_names)
lr_ag_age_per_class_acc_distrib.dropna().to_csv(
    "./results/lr_ag_age_per_class_acc_distrib_GR.csv")
lr_ag_age_per_class_acc_distrib = pd.read_csv(
    "./results/lr_ag_age_per_class_acc_distrib_GR.csv", index_col=0)
lr_ag_age_per_class_acc_distrib = np.round(
    100 * lr_ag_age_per_class_acc_distrib)
lr_ag_age_per_class_acc_distrib_describe = lr_ag_age_per_class_acc_distrib.describe()
lr_ag_age_per_class_acc_distrib_describe.to_csv(
    "./results/lr_ag_age_per_class_acc_distrib_GR.csv")

lr_ag_age_per_class_acc_distrib = pd.melt(
    lr_ag_age_per_class_acc_distrib, var_name="AR age")
lr_ag_age_per_class_acc_distrib
plt.figure(figsize=(4.75, 3))
plt.rc('font', family='Helvetica')
sns.violinplot(x="AR age", y="value", cut=0,
               order=list(lr_ag_age_per_class_acc_distrib["AR age"].unique()),
               data=lr_ag_age_per_class_acc_distrib)
sns.despine(left=True)
plt.xticks(rotation=0, ha="right")
plt.xlabel("Anopheles arabiensis age")
plt.ylabel("Prediction accuracy")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib_GR.pdf",
            bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_per_class_acc_distrib_GR.png",
            bbox_inches="tight")

#%% Confusion matrix
rskf_results = pd.read_csv("./results/lr_ag_age_repeatedCV_record_GR.csv")

best_cm = rskf_results.sort_values(
    by="Accuracy", ascending=False).iloc[0, 4].replace('[ ', '[').split()
best_cm = np.array(ast.literal_eval(",".join(best_cm)))

class_names = y.sort_values().unique()

plt.figure(figsize=(4, 4))
plot_confusion_matrix(best_cm, classes=class_names)
plt.savefig("./plots/lr_ag_age_cm_GR.pdf", bbox_inches="tight")
plt.savefig("./plots/lr_ag_age_cm_GR.png", bbox_inches="tight")
