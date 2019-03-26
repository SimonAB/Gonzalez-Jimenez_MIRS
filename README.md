[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2609356.svg)](https://doi.org/10.5281/zenodo.2609356)

# Prediction of mosquito species and population age structure using mid-infrared spectroscopy and supervised machine learning

Mario González-Jiménez<sup>1*</sup>, Simon A. Babayan<sup>2*</sup>, Pegah Khazaeli<sup>1</sup>, Margaret Doyle<sup>2</sup>, Finlay Walton<sup>1</sup>, Elliott Reedy<sup>1</sup>, Thomas Glew<sup>1</sup>, Mafalda Viana<sup>2</sup>, Lisa Ranford-Cartwright<sup>2</sup>, Abdoulaye Niang<sup>4</sup>, Doreen J. Siria<sup>3</sup>, Fredros O. Okumu<sup>2,3</sup>, Abdoulaye Diabaté<sup>4</sup>, Heather M. Ferguson<sup>2</sup>, Francesco Baldini<sup>2</sup>, and Klaas Wynne<sup>1</sup>

<sup>1</sup> School of Chemistry, University of Glasgow, Glasgow G12 8QQ, UK.\
<sup>2</sup> Institute of Biodiversity Animal Health and Comparative Medicine, University of Glasgow, Glasgow G12 8QQ, UK.\
<sup>3</sup> Environmental Health & Ecological Sciences Department, Ifakara Health Institute, Off Mlabani Passage, PO Box 53, Ifakara, Tanzania\
<sup>4</sup> Department of Medical Biology and Public Health, Institut de Recherche en Science de la Santé (IRSS), Bobo-Dioulasso, Burkina Faso\
<sup>*</sup> These authors contributed equally to this work.
 
### Description
This repository contains code used to process wave number readings, to generate models for the classification of mosquito species (see Fig 2), their ages (Fig 3), and population age structure (Fig 4).

### Files

- `Loco mosquito.ipynb` contains code for processing outputs of the mid-infrared spectrometer into datasets suitable for machine learning
- `OpWT_classification.py` contains a summary of the code used for training classification models for both species and age class prediction.
- `OpWT_population_structure.py` contains code used to compare true and predicted age structures of a simulated population. Predictions were made using the most accurate model trained in `OpWT_classification.py`
