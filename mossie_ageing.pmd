# %% import modules
import os
import glob
import re
from time import time

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, mean_squared_error

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
from sklearn.linear_model import ElasticNet
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

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor

import hdbscan

import matplotlib.pyplot as plt
%matplotlib inline


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

# %% import data
df = pd.read_table("./mosquitoes_spectra.dat", index_col="Age")
df = df.ix[:,1:-1]

# %% transform data
df[df.columns] = StandardScaler().fit_transform(df[df.columns].as_matrix())
df.head()

# %% Spot check classification models

X = df
y = df.index

# cross-val settings
seed = 4
validation_size = 0.30
num_folds = 10

# pick the models to test
models = []
models.append(("LR", LogisticRegression()))
models.append(("SGD", SGDClassifier()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("RF", RandomForestClassifier()))
models.append(("LR", ExtraTreeClassifier()))
models.append(("XGB", xgb.XGBClassifier()))
models.append(("NB", GaussianNB()))
models.append(("LR", SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    cv_results = cross_val_score(model, X, y, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(name, cv_results.mean(), cv_results.std())
    print(msg)

# %% plot
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title('Algorithm comparison (mossie age)')
plt.xticks(rotation=30)
plt.ylabel('Accuracy (median, quartiles, range)')
#plt.show()
plt.savefig(HOME+"/Desktop/spot_check_age.jpeg")

# %% Spot check regression models

X = df
y = df.index

# cross-val settings
seed = 4
validation_size = 0.30
num_folds = 10

# pick the models to test
models = []
models.append(("LR", LogisticRegression()))
models.append(("SGD", SGDClassifier()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("RF", RandomForestClassifier()))
models.append(("LR", ExtraTreeClassifier()))
models.append(("XGB", xgb.XGBClassifier()))
models.append(("NB", GaussianNB()))
models.append(("LR", SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    cv_results = cross_val_score(model, X, y, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(name, cv_results.mean(), cv_results.std())
    print(msg)

# %% plot
sns.boxplot(x=names, y=results)
sns.despine(offset=10, trim=True)
plt.title('Algorithm comparison (mossie age)')
plt.xticks(rotation=30)
plt.ylabel('Accuracy (median, quartiles, range)')
#plt.show()
plt.savefig(HOME+"/Desktop/spot_check_age.jpeg")
