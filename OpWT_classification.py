#%% import modules
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report,\
confusion_matrix,  precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

import pickle

# Set random seed to insure reproducibility
seed = 4

#%% load data
df = pd.read_table("mosquitoes_spectra.dat");

# transform species data
df[df.columns] = StandardScaler().fit_transform(df[df.columns].values)

#%% load data
X = df.values
y = df.index

# cross validation
validation_size = 0.3
num_splits = 10
num_repeats = 10
num_rounds = 10
scoring = "accuracy"

# preparing model
classifier = LogisticRegressionCV(Cs=30,
                                  fit_intercept=True,
                                  cv=10,
                                  dual=False,
                                  penalty="l2",
                                  scoring="accuracy",
                                  solver="lbfgs",
                                  tol=0.0001,
                                  max_iter=1000,
                                  class_weight="balanced",
                                  n_jobs=-1,
                                  verbose=1,
                                  refit=True,
                                  intercept_scaling=1.0,
                                  multi_class="ovr",
                                  random_state=seed)

#%% train
# repeated random stratified splitting of dataset
rskf = RepeatedStratifiedKFold(
    n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

# prepare matrices of results
rskf_results = pd.DataFrame()  # model parameters and global accuracy score
rskf_per_class_results = []  # per class accuracy scores
rkf_scores = pd.DataFrame(
    columns=["species", "scores mean", "scores sem"]).set_index("species")

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
            classifier.scores_)
            # , "Species": df.index.unique()
            })#.set_index("Species")

        # combine score outputs
        rkf_scores = pd.merge(rkf_scores, scores_table,
                              left_index=True, right_index=True, how='outer')

        local_rskf_results = pd.DataFrame([("Accuracy", accuracy_score(y_test, y_pred)),
                                           ("TRAIN", str(train_index)),
                                           ("TEST", str( test_index)),
                                           ("Pred_probas", y_predproba),
                                           ("CM", local_cm),
                                           ("Classification report", local_report),
                                           ("Scores", local_scores),
                                           ("Pickle", pickle.dumps(classifier))]).T

        local_rskf_results.columns = local_rskf_results.iloc[0]
        local_rskf_results = local_rskf_results[1:]
        rskf_results = rskf_results.append(local_rskf_results)

        #per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm) / local_support
        rskf_per_class_results.append(local_acc)

# Results
rskf_results.to_csv("./results/lr_sp_repeatedCV_record.csv", index=False)
rskf_per_class_results.to_csv("./results/rskf_per_class_results.csv", index=False)

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
    "./results/lr_sp_rkf_scores.csv", index=False)

#%% evaluate train models
rskf_results = pd.read_csv("./results/lr_sp_repeatedCV_record.csv")
rkf_scores = pd.read_csv("./results/lr_sp_rkf_scores.csv")

# Per class accuracy
class_names = y.sort_values().unique()
lr_sp_per_class_acc_distrib = pd.DataFrame(rskf_per_class_results,
                                           columns=class_names)
lr_sp_per_class_acc_distrib.dropna().to_csv(
    "./results/lr_sp_per_class_acc_distrib.csv")
lr_sp_per_class_acc_distrib = pd.read_csv(
    "./results/lr_sp_per_class_acc_distrib.csv", index_col=0)
lr_sp_per_class_acc_distrib = np.round(
    100 * lr_sp_per_class_acc_distrib)
lr_sp_per_class_acc_distrib_describe = lr_sp_per_class_acc_distrib.describe()
lr_sp_per_class_acc_distrib_describe.to_csv(
    "./results/lr_sp_per_class_acc_distrib.csv")

