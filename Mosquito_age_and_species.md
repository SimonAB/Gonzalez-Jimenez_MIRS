# Predicting  both age and species of *Anopheles gambiae* and *Anopheles arabiensis* from near-infrared spectra

## Statistical learning

Many algorithms have been developed over the years. Because it is very hard to
predict which one will do justice to the specific data we have, it is useful to
try out representative members of the big 'families' of algorithms, e.g. those
based on linear regression, nearest neighbours, decisions trees, or bayesian.

Algorithms used here are:

- LR: logistic regression
- SGD: stochastic gradient descent
- KNN: k nearest neighbours
- CART: classification and regression trees
- RF: random forests
- ET: extra trees
- XGB: extreme gradient boosting
- NB: gaussian naive bayes
- SVM: support vector machines

### Predicting species and age at the same time

This concatenates species and age as a single label.

#### Spot-checking baseline performance of various algorithms

With output category consisting of ages [1, 3, 5, 7, 9, old], XGB achieved the
best prediction accuracy at baseline settings:

![Spotchecking age](plots/spot_check_species_age_rus.png)


#### After tuning XGBoost parameters

Accuracy on test set:47.83%

Classification report:

|Species_Age| precision | recall | f1-score | support|
|:-----------|:----------|:-------|:---------|:-------|     
|AG_1        | 0.58      | 0.78   | 0.67     | 9|       
|AG_3        | 0.36      | 0.62   | 0.46     | 13|      
|AG_5        | 0.50      | 0.81   | 0.62     | 16|      
|AG_7        | 0.45      | 0.38   | 0.42     | 13|      
|AG_old      | 0.45      | 0.28   | 0.34     | 18|      
|AR_1        | 0.88      | 0.41   | 0.56     | 17|      
|AR_3        | 0.40      | 0.67   | 0.50     | 9|       
|AR_5        | 0.44      | 0.31   | 0.36     | 13|      
|AR_7        | 0.47      | 0.41   | 0.44     | 17|      
|AR_old      | 0.44      | 0.31   | 0.36     | 13|      
|avg / total | 0.51      | 0.48   | 0.47     | 138|     

#### Confusion matrix

![Confusion matrix Species_Age](plots/xgb_CM_species_age_rus.png)

#### Top features

Three wavelengths stood out as being particularly important to the prediction:
['1900.76462', '3855.53371', '1745.50175'].

Ranked by decreasing importance:

![Feature importances](plots/xgb_feat_imp_species_age_rus.png)

### Predicting species only
This uses the binary label for species (AG or AR).

#### Spot-checking baseline performance of various algorithms

With output category consisting of ages [1, 3, 5, 7, 9, old], XGB achieved the
best prediction accuracy at baseline settings:

![Spotchecking age](plots/spot_check_species_rus.png)

Both random forest and xgboost performed well here.

#### After tuning Random Forest parameters

Accuracy on test set:84.67%

Classification report:

|             | precision | recall | f1-score | support |
|:------------|:----------|:-------|:---------|:--------|
| AG          | 0.82      | 0.88   | 0.85     | 255     |
| AR          | 0.88      | 0.82   | 0.84     | 267     |
| avg / total | 0.85      | 0.85   | 0.85     | 522     |

#### Confusion matrix

![Confusion matrix Species_Age](plots/RF_CM_species_rus.png)


#### After tuning XGBoost parameters

Accuracy on test set:85.25%

Classification report:

|             | precision | recall | f1-score | support |
|:------------|:----------|:-------|:---------|:--------|
| AG          | 0.84      | 0.86   | 0.85     | 255     |
| AR          | 0.87      | 0.84   | 0.85     | 267     |
| avg / total | 0.85      | 0.85   | 0.85     | 522     |

#### Confusion matrix

![Confusion matrix Species_Age](plots/xgb_CM_species_rus.png)

#### Top features

Four wavelengths stood out as being particularly important to the prediction:
'525.57926', '3855.53371', '1900.76462', '1028.97811'

Ranked by decreasing importance:

![Feature importances](plots/xgb_feat_imp_species_rus.png)

### Predict age from both species

#### Spot-checking baseline performance of various algorithms

With output category consisting of ages [1, 3, 5, 7, 9, old], XGB achieved the
best prediction accuracy at baseline settings:

![Spotchecking age](plots/spot_check_age_rus_AGandAR.png)

#### After tuning XGBoost parameters

#### Confusion matrix

![Confusion matrix Species_Age](plots/xgb_CM_age_rus_AGandAR.png)

#### Top features

Ranked by decreasing importance:

![Feature importances](plots/xgb_feat_imp_age_rus_AGandAR.png)


Four wavelengths stood out as being particularly important to the prediction:
'3855.53371', '1900.76462', '1745.50175', '2922.99216'


### Predicting age separately for AG ad AR
I then built 2 separate models of age: one selecting only AG and the other with only AR.

#### Predicting age of AG

#### Spot-checking baseline performance of various algorithms

With output category consisting of ages [1, 3, 5, 7, 9, old], XGB achieved the
best prediction accuracy at baseline settings:

![Spotchecking age](plots/spot_check_age_rus_AG.png)

#### After tuning XGBoost parameters

#### Confusion matrix

![Confusion matrix Species_Age](plots/xgb_CM_age_rus_AG.png)

#### Top features

Ranked by decreasing importance:

![Feature importances](plots/xgb_feat_imp_age_rus_AG.png)


Four wavelengths stood out as being particularly important to the prediction:



#### Predicting age of AR

#### Spot-checking baseline performance of various algorithms

With output category consisting of ages [1, 3, 5, 7, 9, old], XGB achieved the
best prediction accuracy at baseline settings:

![Spotchecking age](plots/spot_check_age_rus_AR.png)

#### After tuning XGBoost parameters

#### Confusion matrix

![Confusion matrix Species_Age](plots/xgb_CM_age_rus_AR.png)

#### Top features

Ranked by decreasing importance:

![Feature importances](plots/xgb_feat_imp_age_rus_AR.png)


Three wavelengths stood out as being particularly important to the prediction:
'1900.76462', '1745.50175', '3855.53371'

### Conclusions

1. Predicting age and species at the same time yields an accuracy of **47%**.
2. However, using the full dataset (which includes *Anopheles gambiae* and *Anopheles arabiensis*), to predict species alone achieves **85.25%** accuracy (xgboost)
3. predicting age using both AG and AR achieves **52%** accuracy
4. predicting age using AG only achieves **71%** accuracy
5. predicting age using AR only achieves **71.5%** accuracy