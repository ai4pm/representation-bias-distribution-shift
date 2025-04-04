# Data Inequality and Distribution Shifts on Multi-Ancestral Machine Learning for Polygenic Prediction
Used AMR, SAS, EAS, and AFR population


# Software description
Here is the software description implementing Multi-Ancestral Machine Learning for Polygenic Prediction

## Entity Path/location Note
**Data:** ./10000P_500SNPs/
- common_snps_ids.csv (contains the common SNPs across Multi-Ancestry)
- data_???.csv (Contains common 500 SNPs for five different ancestry, extracted from Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/COXHAP)
- data_???_flipped.csv (contains same coding of alleles with respect to EUR's major alleles and minor alleles)

**Data Extraction:** To extract above data do following: 
- Download Harvard Dataverse (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/COXHAP)
- run: ./Synthetic_code_finalcode_final/1_Extraction_Similar_SNPs_for_all_ancestory.py
- then run: ./Synthetic_code_finalcode_final/2_Extracted_data_EDA_Analysis.py

**Model:** ./Synthetic_code_finalcode_final/
- 3_neural_network_exisitng_Layernorm.py:  This interface produces Polygenic Predictions of Multi-Ancestry
- In this REGRESSION_FLAG controls wheather we want regression or classification. for classification set REGRESSION_FLAG = False.
- /genomic_risk_case_control.py It generates binary case control for the cohorts for clasification task. here np.percentile controls the percentage of case-s-controls.
- /genomic_risk_case_control_regression.py It generates continuous phenotypes
- /5_Raw_pred_to_Scores.py It can be used to generate different evaluation metric from raw predictions
- /4_Medain_values_of_experiments.py It can be used to generate excel file with 4 or 5 sheet, comparing EUR Vs DDP.
  

