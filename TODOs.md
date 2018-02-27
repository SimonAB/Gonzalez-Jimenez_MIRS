# TODO

- [x] prediction of species and age
- [x] prediction of species
- [x] prediction of Arabiensis age
- [x] prediction of Anopheles age
- [x] complete report

## meeting 16 January 2018

- [x] use only classification approach for age predictions from now on, with label as 'real age' (1-17)
- [ ] try XGB to better distinguish consecutive days
- [ ] merge days in output if I cannot disentangle them: use mean of 2 or 3 consequtive days to match other species
- [ ] send matrix of prediction probabilities and confusion matrices.
- [ ] use 'wild mossie' dataset to test models trained on lab mossies.
- [ ] fix LR for species classification

## Meeting 27 Feb 2018

- [ ] Mario and Francesco will double check the dataset (during the meeting we observed a potentiall error in a few labeling)
- [ ] use new dataset including D3 AG 
- [ ] Construct validation set with missing days
- [ ] combine SF+GR for training & validation
- [ ] if not look good for old mossies: [11; 15] 
- [ ] Mafalda and Simon will work on figure 4 using our best model