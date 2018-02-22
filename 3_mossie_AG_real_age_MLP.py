# %% Load modules
import numpy as np
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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
X = df.values

y = df.index

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

print("X:",X.shape, "\ny:", len(np.unique(y)))

# %% define models
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(17, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def deeper_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=16, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(6, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(17, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=16, kernel_initializer='normal', activation='relu')) 
    model.add(Dense(17, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 4
numpy.random.seed(seed)

# %% evaluate model
estimator = KerasClassifier(build_fn=baseline_model,
                            epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" %
      (results.mean() * 100, results.std() * 100))
