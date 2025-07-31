import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import os

# Parameters
h_sq_list = [0.25, 0.5]
rho_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
lambda__list = [0.1, 0.2, 0.3, 0.4, 0.5]
metric_prefix = '_sq_corr'
dict_learn = ['Mix0' + metric_prefix, 'Mix1' + metric_prefix, 'Mix2' + metric_prefix, 'ind_1' + metric_prefix, 'ind_2' + metric_prefix, 'naive' + metric_prefix, 'tl2' + metric_prefix]
ddp_list = ['eur', 'amr', 'sas', 'eas', 'afr']

output_dir = 'C:/Users/skumar26/synthetic_data_Generation_RESULTS/Gaps_plots'
os.makedirs(output_dir, exist_ok=True)

for temp_file_prefix in [
    # 'neural_10000P_100SNPs_regression',
    # 'XGBoost_10000P__100SNPs_regression',
    # 'tabtransformer_regression'
    # 'neural_classification_10000P_500SNPs'
    'neural_regression_random_genomic_val_test_together_40_runs_3000SNPs_100000_patients'
    ]: 
    print('Read file:', temp_file_prefix)
    for h_sq in h_sq_list:
        output_file = os.path.join(output_dir, f'{temp_file_prefix}_h_Sq_{h_sq}_results.xlsx')
        results = {ddp_str: pd.DataFrame(columns=['lambda', 'rho'] + dict_learn) for ddp_str in ddp_list}
        for ddp_str in ddp_list:
            for lambda_ in lambda__list:
                for rho in rho_list:
                    file_name = f'h_sq_{h_sq}_rho_{rho}_lambda_{lambda_}{ddp_str}'
                    file_path = f'C:/Users/skumar26/synthetic_data_Generation_RESULTS/{temp_file_prefix}/{ddp_str}/COMPILED_{file_name}.csv'
                    # Read file if it exists
                    if not os.path.exists(file_path):
                        print(f'Error: file does not exist: {file_path}')
                        exit(1)
                    else:
                        df = pd.read_csv(file_path)
                    # Check if dataframe has 40 rows
                    if len(df) != 40:
                        print(f'Error: length of df is not 40 in file {file_path}')
                        exit(1)

                    all_medians = []
                    for key in dict_learn:           
                        # Replace any negative value with 0 in the key column
                        # df[key] = df[key].clip(lower=0)
                        # Calculate the median of the data
                        median_gap = df[key].median()
                        all_medians.append(median_gap)
                    # Concatenate all medians for the given lambda and rho
                    median_data = pd.DataFrame({
                        'lambda': [lambda_],
                        'rho': [rho],
                        **{key: [median] for key, median in zip(dict_learn, all_medians)}
                    })
                    results[ddp_str] = pd.concat([results[ddp_str], median_data], ignore_index=True)
                    del df

        # Before saving, let's modify column names if needed
        column_name_mapping = {'ind_1': 'Ind1', 'ind_2': 'Ind2', 'naive': 'NT', 'tl2': 'TL'}  
        results = {ddp_str: result_df.rename(columns=column_name_mapping) for ddp_str, result_df in results.items()}
        
        # Check and correct sheet names if necessary
        sheet_name_mapping = {'amr': 'EUR + AMR', 'sas': 'EUR + SAS', 'eas': 'EUR + EAS', 'afr': 'EUR + AFR', 'eur': 'EUR+EUR'}

        # Save results with the corrected sheet names
        with pd.ExcelWriter(output_file) as writer:
            for ddp_str, result_df in results.items():
                # result_df.sort_values(by=['lambda', 'rho'], inplace=True)
                sheet_name = sheet_name_mapping.get(ddp_str, ddp_str)
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)

# Now load the saved Excel file and display a summary
saved_results = pd.ExcelFile(output_file)
saved_sheets_data = {sheet: saved_results.parse(sheet) for sheet in saved_results.sheet_names}
saved_sheets_summary = {sheet: data.head() for sheet, data in saved_sheets_data.items()}

print(saved_sheets_summary)
