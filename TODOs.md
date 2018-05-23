# TODO

- [x] prediction of species and age
- [x] prediction of species
- [x] prediction of Arabiensis age
- [x] prediction of Anopheles age
- [x] complete report

## meeting 16 January 2018

- [x] use only classification approach for age predictions from now on, with label as 'real age' (1-17)
- [x] merge days in output if I cannot disentangle them: use mean of 2 or 3 consecutive days to match other species
- [x] try XGB to better distinguish consecutive days
- [x] send matrix of prediction probabilities and confusion matrices.
- [x] use 'wild mossie' dataset to test models trained on lab mossies.
- [x] fix LR for species classification

## Meeting 27 Feb 2018

- [x] Mario and Francesco will double check the dataset (during the meeting we observed a potential error in a few labelling)
- [x] use new dataset including D3 AG
- [x] combine SF+GR for training & validation
- [x] Construct validation set 1 with missing days
- [x] Construct validation set 2 with population's age structure
- [x] if not look good for old mossies: [11; 15]
- [x] use average CMs over 10 top models
- [x] Mafalda and Simon will work on figure 4 using our best model: ask mafalda for intervention in arabiensis
- [ ] do stats comparison between predicted naive and int popualtions (KS test?)
- [x] use same 3-day intervention process for arabiensis (Mafalda)
- [x] run the arabiensis models
- [ ] test effects of storage time, etc on the top predictors