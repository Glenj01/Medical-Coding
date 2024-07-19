# Explainable Automated Medical Coding

This is the code utilised in the study A Comparative Study on Automatic Coding of Medical Letters with Explainability, Jamie Glen and Lifeng Han and Paul Rayson and Goran Nenadic, 2024, 2407.13638, arXiv, cs.CL, https://arxiv.org/abs/2407.13638 

I would like to acknowledge that the pre-trained model, and some of the code used to run it, is from the paper Dong, H., Suárez-Paniagua, V., Whiteley, W., Wu, H. (2021) Explainable automated coding of clinical notes using hierarchical label-wise attention networks and label embedding initialisation, Journal of Biomedical Informatics, Volume 116, 2021, 103728, ISSN 1532-0464. https://doi.org/10.1016/j.jbi.2021.103728 

# Requirements
* Python 3.6
* Tensorflow 1.15
* [Numpy](http://www.numpy.org/)
* [scikit-learn](http://scikit-learn.github.io/stable) for implementing evaluation metrics
* [Gensim](https://radimrehurek.com/gensim/) 3.* (tested on 3.8.3) for pre-training word and label embeddings with the word2vec algorithm
* [NLTK](https://www.nltk.org/) for tokenisation
* [Spacy](https://spacy.io/) 2.3.2 (before 3.x) for customised rule-based sentence parsing
* [TFLearn](http://tflearn.org/) for sequence padding

# Resources
* [HLAN model](https://github.com/acadTags/Explainable-Automated-Medical-Coding).
* [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) Follow the instructions on this page to download the full database
* [Snomed Mappings](https://www.nlm.nih.gov/research/umls/mapping_projects/icd9cm_to_snomedct.html#:~:text=The%20most%20useful%20maps%20are,to%2Done%20maps%20as%20possible) Dec 2022 release.

# To run with MIMIC-III ICD Coding and replicate the studys results
* Clone the HLAN github repo onto your device, then replace any duplicate folders from that with folders from this repo. This is all the folders aside from ../caml-preprocessing. 
* Download the full MIMIC database, then follow the README in the folder ../caml-preprocessing to preprocess the data.
* Either train the model or download the pretrained model embeddings. Instructions for either can be found on the HLAN github (https://github.com/acadTags/Explainable-Automated-Medical-Coding).
* Download the ICD to SNOMED maps and rename them ICD_SNOMED_1TO1 and ICD_SNOMED_1TOM accordingly, placing them in the ../HLAN folder.
* Place the medical document in the 'noteInput.txt' folder and run the 'runTest.py' program in ../HLAN. This will output its predicted results and return the embeddings in the ../embeddings folder. 

# Cite us
@misc{glen2024comparativestudyautomaticcoding, title={A Comparative Study on Automatic Coding of Medical Letters with Explainability}, author={Jamie Glen and Lifeng Han and Paul Rayson and Goran Nenadic}, year={2024}, eprint={2407.13638}, archivePrefix={arXiv}, primaryClass={cs.CL}, url={https://arxiv.org/abs/2407.13638}, }

