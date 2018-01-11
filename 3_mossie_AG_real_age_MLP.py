# %% Load modules
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd
import matplotlib.pyplot as plt

#%% import data
df_real_age = pd.read_table("./mosquitoes_data (2018).dat", index_col="Age")
df_AG_real_age = df_real_age[df_real_age["Species"] == "AG"]
df_AG_real_age = df_AG_real_age.iloc[:, 3:-1]
df_AG_real_age.head()

df_AG_real_age[df_AG_real_age.columns] = StandardScaler().fit_transform(
    df_AG_real_age[df_AG_real_age.columns].as_matrix())
df_AG_real_age.head()

df = df_AG_real_age.copy()
y = df.index
X = df

print(len(X.columns))

# %% define models
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(6, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=16, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 4
numpy.random.seed(seed)

# %% evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0) # if using no standardisation

# evaluate model with standardised dataset
estimators = []
estimators.append(('standardise', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
# estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
# results = cross_val_score(estimator, X, y, cv=kfold)
print("MSE: %.2f ± %.2f" % (results.mean(), results.std()))
