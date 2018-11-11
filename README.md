# GAMENet
GAMENet : Graph Augmented MEmory Networks for Recommending Medication Combination

For reproduction of medication prediction results in our [paper](https://arxiv.org/abs/1809.01852), see instructions below.

## Overview
This repository contains code necessary to run GAMENet model. GAMENet is an end-to-end model mainly based on graph convolutional networks (GCN) and memory augmented nerual networks (MANN). Paitent history information and drug-drug interactions knowledge are utilized to provide safe and personalized recommendation of medication combination. GAMENet is tested on real-world clinical dataset [MIMIC-III](https://mimic.physionet.org/) and outperformed several state-of-the-art deep learning methods in heathcare area in all effectiveness measures and also achieved higher DDI rate reduction from existing EHR data.


## Requirements
- Pytorch >=0.4
- Python >=3.5


## Running the code
### Data preprocessing
In ./data, you can find the well-preprocessed data in pickle form. Also, it's easy to re-generate the data as follows:
1.  download [MIMIC data](https://mimic.physionet.org/gettingstarted/dbsetup/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/
2.  download [DDI data](https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0) and put it in ./data/
3.  run code **./data/EDA.ipynb**

Data information in ./data:
  - records_final.pkl is the input data with four dimension (patient_idx, visit_idx, medical modal, medical id) where medical model equals 3 made of diagnosis, procedure and drug.
  - voc_final.pkl is the vocabulary list to transform medical word to corresponding idx.
  - ddi_A_final.pkl and ehr_adj_final.pkl are drug-drug adjacency matrix constructed from EHR and DDI dataset.
  - drug-atc.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt are mapping files for drug code transformation.
  
  
### Model Comparation
 Traning codes can be found in ./code/baseline/
 
 - **Nearest** will simply recommend the same combination medications at previous visit for current visit.
 - **Logistic Regression (LR)** is a logistic regression with L2 regularization. Here we represent the input data by sum of one-hot vector. Binary relevance technique is used to handle multi-label output.
 - **Leap** is an instance-based medication combination recommendation method.
 - **RETAIN** can provide sequential prediction of medication combination based on a two-level neural attention model that detects influential past visits and significant clinical variables within those visits.
 - **DMNC** is a recent work of medication combination prediction via memory augmented neural network based on differentiable neural computers. 
 
 
 ### GAMENet
 ```
 python train_GAMENet.py --model_name GAMENet --ddi# training with DDI knowledge
 python train_GAMENet.py --model_name GAMENet --ddi --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge
 python train_GAMENet.py --model_name GAMENet # training without DDI knowledge
 python train_GAMENet.py --model_name GAMENet --resume_path Epoch_{}_JA_{}_DDI_{}.model --eval # testing with DDI knowledge
 ```
 
## Cite 

Please cite our paper if you use this code in your own work:

```
@article{shang2018gamenet,
  title="{GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination}",
  author={Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  journal={arXiv preprint arXiv:1809.01852},
  year={2018}
}
```
