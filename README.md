# Learning Survival Distributions with Individually Calibrated Asymmetric Laplace Distribution (ICLR 2026)

This repository contains the official implementation of our ICML 2025 paper:

**Learning Survival Distributions with Individually Calibrated Asymmetric Laplace Distribution** 📄

## 🔍 Overview
Survival analysis plays a critical role in modeling time-to-event outcomes across various domains. 
Although recent advances have focused on improving *predictive accuracy* and *concordance*, fine-grained *calibration* remains comparatively underexplored. 
In this paper, we propose a survival modeling framework based on the Individually Calibrated Asymmetric Laplace Distribution (ICALD), which unifies *parametric* and *nonparametric* approaches based on the ALD.
We begin by revisiting the probabilistic foundation of the widely used *pinball* loss in *quantile regression* and its reparameterization as the *asymmetry form* of the ALD. 
This reparameterization enables a principled shift to *parametric* modeling while preserving the flexibility of *nonparametric* methods.
Furthermore, we show theoretically that ICALD, with the *quantile regression* loss is probably approximately individually calibrated.
Then we design an extended ICALD framework that supports both *pre-calibration* and *post-calibration* strategies. 
Extensive experiments on 14 synthetic and 7 real-world datasets demonstrate that our method achieves competitive performance in terms of *predictive accuracy*, *concordance*, and *calibration*, while outperforming 12 existing baselines including recent *pre-calibration* and *post-calibration* methods.


## 🗂️ Directory Structure
```
datasets/
├── breast_msk_2018_clinical_data.tsv   
├── gbsg_cancer_train_test.h5            
├── gbsg.csv                             
├── lgggbm_tcga_pub_clinical_data.tsv    
├── metabric_IHC4_clinical_train_test.h5  
├── support_train_test.h5                
├── tmb_immuno_mskcc.tsv                 
└── whas_train_test.h5                  # real-world data with real censoring datasets

figures/
├── *.png                               # Figures and visualizations generated during analysis

res/
├── *.json                              # Experiment results

./
├── datasets.py                         # Functions for loading and preprocessing datasets
├── hyperparams.py                      # Hyperparameters used in the experiments 
├── models.py                           # Neural network architectures and custom losses in the experiments 
├── script.py                           # Main script for running experiments
├── utils.py                            # Utility functions for common operations

requirements.txt                        # Python dependencies
README.md                               # Project description and usage instructions
```

---

## **Requirements**
Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Data Preparation**
Place all datasets in the `datasets/` folder. Preprocessing functions are available in `datasets.py`.

### **2. Running Experiments**
Use `script.py` to train and evaluate models:
```bash
python script.py
```

### **3. Visualizations**
Figures and plots generated during analysis will be saved in the `figures/` folder.


📌 Note: This paper has been accepted to ICLR 2026. The official proceedings citation will be updated once available.
