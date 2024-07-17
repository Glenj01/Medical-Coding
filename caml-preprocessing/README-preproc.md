# caml-mimic

Code for the paper [Explainable Prediction of Medical Codes from Clinical Text](https://arxiv.org/abs/1802.05695).

## Dependencies
* Python 3.6, though 2.7 should hopefully work as well
* pytorch 0.3.0
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.20.3
* gensim 3.2.0
* nltk 3.2.4

Other versions may also work, but the ones listed are the ones I've used

## Data processing

The MIMIC-III Database is required for this preprocessing to work, specifically the NOTEVENTS, DIAGNOSES_ICD, PROCEDURES_ICD, D_DIAGNOSES_ICD, and D_PROCEDURES_ICD. 

To get started, first edit `constants.py` to point to the directories holding your copies of the MIMIC-III dataset. Then, organize your data with the following structure:
```
mimicdata
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions (already in repo)
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (already in repo)
```

Now, make sure your python path includes the base directory of this repository. Then, run the program`notebooks/dataproc_mimic_III.py`.