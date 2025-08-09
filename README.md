# Data Representation Bias and Conditional Distribution Shift Drive Predictive Performance Disparities in Multi-Population Machine Learning


# Software description
Here is the software description implementing Multi-Ancestral Machine Learning for Polygenic Prediction

## Data: To extract the data, do the following: 
  
| Step   | Path/Link | Note  |
|--------|------|------------|
| 1      |https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/COXHAP |Download Harvard Dataverse. Duplicate chr1.fam file for remaining chromosomes. |
| 2      | https://pypi.org/project/bed-reader/ | pip install bed-reader |
| 3      |./Data/Extraction_Similar_SNPs_for_all_ancestory.py | To extract data for all the ancestry as in the manuscript, set variables: n (no. of patients) = 10000; m (no. of SNPs) = 500;|
| 4      |  ./Data/Extracted_data_EDA_Analysis.py  |This ensures that coding of alleles in all ancestries are in synchronus to european ancestry       |
## Examples  ./examples/
- common_snps_ids.csv (contains the common SNPs across Multi-Ancestry)
- data_*.csv (Contains common 500 SNPs for five different ancestry, extracted from Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/COXHAP)
- data_*_flipped.csv (contains same coding of alleles with respect to EUR's major alleles and minor alleles)

## Polygenic scores generation
./Polygenic scores generation/
- /genomic_risk_case_control.py:  It generates binary case control for the cohorts for clasification task. Here np.percentile controls the percentage of case-controls.
- /genomic_risk_case_control_regression.py:  It generates continuous phenotypes

## Model:
./model/
- /neural_network_exisitng_Layernorm.py:  This interface produces Polygenic Predictions of Multi-Ancestry using keras framework. In this REGRESSION_FLAG controls if we want regression or classification. For classification assign REGRESSION_FLAG = False.


## results_post_processing
./results_post_processing/
- /Raw_pred_to_Scores.py:  It can be used to generate different evaluation metric from raw predictions
- /Medain_values_of_experiments.py:  It can be used to generate excel file (based on different metrics) with 4 or 5 sheet, where each sheet compares EUR Vs a DDP.

## shell_file_creation_for_cluster
./shell_file_creation_for_cluster/
- /H100_GPU_shell_file_creation_h_sq_rho_seperate.py:  It generates numerous shell files to run a seperate /model/neural_network_exisitng_Layernorm.py for each of the required combinations of rho, heritability and DDP on H100 GPUs
- /automated_shell_file_creation_h_sq_rho_seperate.py:  It does the same task as above for a CPU cluster

## Libraries required:
- Python==3.11.9
- graphviz==0.20.1
- h5py==3.11.0
- imbalanced-learn==0.12.3
- keras==3.5.0
- keras-tuner==1.4.7
- matplotlib==3.7.1
- matplotlib-inline==0.1.6
- numpy==1.23.5
- pandas==2.0.1
- scikit-learn==1.2.2
- scipy==1.10.1
- seaborn==0.12.2
- tensorflow==2.17.0

## Authors:
Sandeep Kumar and Yan Cui, ({skumar26, ycui2}@uthsc.edu).
  

