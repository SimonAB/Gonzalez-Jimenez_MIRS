# Predicting mosquito age from near-infrared spectra
## Statistical learning
### Using full age ranges.
### spot-checking baseline performance of various algorithms
With output category consisting of ages [1, 3, 5, 7, 9, 11, 13, 15, 17], prediction accuracy was quite poor at baseline settings:
![Spotchecking age](plots/spot_check_age.png)

### After tuning XGBoost parameters
Accuracy on test set:34.98%

Classification report:

|             | precision | recall | f1-score | support |
|:------------|:----------|:-------|:---------|:--------|
|             |           |        |          |         |
| 1           | 0.24      | 0.50   | 0.32     | 8       |
| 3           | 0.48      | 0.48   | 0.48     | 31      |
| 5           | 0.27      | 0.31   | 0.29     | 29      |
| 7           | 0.42      | 0.27   | 0.33     | 37      |
| 9           | 0.34      | 0.37   | 0.35     | 41      |
| 11          | 0.30      | 0.23   | 0.26     | 30      |
| 13          | 0.28      | 0.23   | 0.25     | 22      |
| 15          | 0.00      | 0.00   | 0.00     | 14      |
| 17          | 0.47      | 0.65   | 0.54     | 31      |
| avg / total | 0.35      | 0.35   | 0.34     | 243     |
