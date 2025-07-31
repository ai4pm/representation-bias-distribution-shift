from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, auc, precision_recall_curve
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
import os
REGRESSION_FLAG = True
def calculate_R2(y_true, y_pred):
    # Calculate the mean of the true labels
    y_mean = np.mean(y_true)
    
    # Calculate the total sum of squares
    ss_total = np.sum((y_true - y_mean) ** 2)
    
    # Calculate the residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Calculate the R^2 score
    R2 = 1 - (ss_res / ss_total)
    
    return R2

def calculate_squared_correlation(y_true, y_pred):
    assert not np.any(np.isnan(y_true)), "y_true contains NaN values"
    assert not np.any(np.isnan(y_pred)), "y_pred contains NaN values"
    assert not np.any(np.isinf(y_true)), "y_true contains inf or -inf values"
    assert not np.any(np.isinf(y_pred)), "y_pred contains inf or -inf values"
    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        print("y_true or y_pred has zero variance")
        return 0
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    corr_squared = corr ** 2
    assert not np.isnan(corr_squared), "corr_squared is NaN"
    assert not np.isinf(corr_squared), "corr_squared is inf or -inf"
    return corr_squared


def calculate_SMAPE(y_true, y_pred):
    # Calculate the absolute percentage error for each prediction
    APE = np.abs((y_true - y_pred)) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    # Calculate the mean absolute percentage error
    SMAPE = np.mean(APE) * 100
    
    return SMAPE

def calculate_log_likelihood(labels, probabilities):
    # Ensuring the probabilities are clipped to avoid log(0) which is undefined
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    log_likelihood = np.sum(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
    return log_likelihood

def calculate_tjur_R2(y_true, y_pred_probs):
    mean_prob_1 = np.mean(y_pred_probs[y_true == 1])
    mean_prob_0 = np.mean(y_pred_probs[y_true == 0])

    assert len(y_pred_probs[y_true == 1]) > len(y_true)/2.15
    assert len(y_pred_probs[y_true == 0]) > len(y_true)/2.15

    R2_tjur = mean_prob_1 - mean_prob_0
    
    return R2_tjur

def calculate_nagelkerkes_r2(y_observed, y_predicted, n):
    y_null = y_observed.mean()
    log_likelihood = calculate_log_likelihood(y_observed, y_predicted)
    #print(f"The log-likelihood of the model is: {log_likelihood}")
    log_likelihood_null = calculate_log_likelihood(y_observed, y_null)
    #print(f"The log-likelihood of the null model is: {log_likelihood_null}")
    R2_McFadden = 1 - (log_likelihood) / log_likelihood_null
    print(f'McFadden R^2: {R2_McFadden}')
    R2_cox_snell = 1 - (np.exp(log_likelihood_null*2/n) / np.exp(log_likelihood*2/n))
    print(f'Cox and Snell R^2: {R2_cox_snell}')
    R2_nagelkerke = R2_cox_snell / (1 - np.exp(log_likelihood_null*2/n))
    return R2_nagelkerke


def calculate_ppv_npv(y_true, y_scores):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Calculate Youden's index to find the optimal threshold
    youdens_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youdens_index)]
    
    # Apply the optimal threshold to classify predictions
    y_pred_classified = (y_scores >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classified).ravel()
    
    # Calculate PPV and NPV
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return ppv, npv


# Prepare a list to hold all the results

h_sq_list = [0.25, 0.5]
rho_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
lambda__list = [0.1, 0.2, 0.3, 0.4, 0.5]
for temp_file_prefix in ['neural_regression_random_genomic_val_test_together_40_runs_3000SNPs_100000_patients']:
    print(temp_file_prefix)
    for h_sq in h_sq_list:
        for ddp_str in ['eur','amr', 'sas', 'eas', 'afr']:
            for rho in rho_list:
                for lambda_ in lambda__list:
                    results = []
                    title = f'h_sq_{h_sq}_rho_{rho}_lambda_{lambda_}{ddp_str}'
                    path_temp = 'C:/Users/skumar26/synthetic_data_Generation_RESULTS/'+ temp_file_prefix+f'/{ddp_str}/'
                    file_name =  path_temp+'pred_' + title + '.csv'
                    df1 = pd.read_csv(file_name)
                    # print(df1.head())
                    assert len(df1['seed'].unique()) == 40
                    for seed in df1['seed'].unique():
                        print(f'Processing seed {seed}...')
                        temp_df = df1[df1['seed'] == seed].drop(columns='seed')
                        if ddp_str == 'eur':
                            eur_ind_count = temp_df[temp_df['info'] == 'eur_IND'].shape[0]
                            assert eur_ind_count == 2, f'DDP_str EUR_IND count is {eur_ind_count}'
                            temp_df.loc[temp_df[temp_df['info'] == 'eur_IND'].index[1], 'info'] = '2' + temp_df.loc[temp_df[temp_df['info'] == 'eur_IND'].index[1], 'info']
                        # Initialize a dictionary to store the results for the current seed
                        seed_results = {'seed': seed}
                        
                        # Define scenarios
                        scenarios = ['Mix0', 'Mix1', 'Mix2', 'ind_1', 'ind_2', 'naive', 'tl2']
                        
                        # Loop through scenarios to calculate metrics
                        for scenario in scenarios:
                            row_temp = temp_df[temp_df['info'] == 'R_test']
                            if scenario in ['Mix1', 'ind_1']:  # Use EUR data
                                cols_temp = row_temp.columns[(row_temp != 0).any()]
                                ancestry = 'eur'
                            elif scenario in ['Mix2', 'ind_2', 'naive', 'tl2']:  # Use AFR data
                                cols_temp = row_temp.columns[(row_temp == 0).any()]
                                ancestry = ddp_str
                                if ddp_str == 'eur':
                                    ancestry = '2eur'
                            else:  # Mix0 doesn't need special handling
                                cols_temp = temp_df.columns
                            del row_temp
                            
                            if 'info' not in cols_temp:
                                cols_temp = np.append(cols_temp, 'info')
                            # assert last column is 'info'
                            assert cols_temp[-1] == 'info'

                            temp_df_scenario = temp_df[cols_temp]
                            # print(temp_df_scenario.head())
                            if scenario == 'Mix0':
                                if ddp_str == 'eur':
                                    assert df1[df1['seed'] == seed].drop(columns=['seed', 'info']).equals(temp_df_scenario.drop(columns='info'))
                                else:
                                    assert df1[df1['seed'] == seed].drop(columns='seed').equals(temp_df_scenario)
                            
                            # Extract predictions and true labels based on the scenario
                            if scenario in ['Mix0', 'Mix1', 'Mix2']:
                                y_pred = temp_df_scenario[temp_df_scenario['info'] == 'Mix0'].iloc[:, :-1].values.reshape(-1)
                            else:
                                if scenario in ['ind_1', 'ind_2']:
                                    y_pred = temp_df[temp_df['info'] == ancestry+'_IND'].iloc[:, :-1].values.reshape(-1)
                                else:
                                    y_pred = temp_df[temp_df['info'] == scenario].iloc[:, :-1].values.reshape(-1)
                                y_pred = y_pred[~np.isnan(y_pred)]
                            
                            # Calculate metrics
                            y_labels = temp_df_scenario[temp_df_scenario['info'] == 'Y_test'].iloc[:, :-1].values.reshape(-1)
                            if REGRESSION_FLAG:
                                R2 = calculate_R2(y_labels, y_pred)
                                SMAPE = calculate_SMAPE(y_labels, y_pred)
                                squared_corr = calculate_squared_correlation(y_labels, y_pred)

                                # Store results
                                seed_results[f'{scenario}_R2'] = R2
                                seed_results[f'{scenario}_SMAPE'] = SMAPE
                                seed_results[f'{scenario}_sq_corr'] = squared_corr

                            else:
                                auc = roc_auc_score(y_labels, y_pred)
                                precision, recall, _ = precision_recall_curve(y_labels, y_pred)
                                aupr = auc(recall, precision)
                                ppv, npv = calculate_ppv_npv(y_labels, y_pred)
                                tjur_R2 = calculate_tjur_R2(y_labels, y_pred)
                                nagelkerkes_r2 = calculate_nagelkerkes_r2(y_labels, y_pred, len(y_labels))
                                
                                # Store results
                                seed_results[f'{scenario}_AUC'] = auc
                                seed_results[f'{scenario}_PR'] = aupr
                                seed_results[f'{scenario}_PPV'] = ppv
                                seed_results[f'{scenario}_NPV'] = npv
                                seed_results[f'{scenario}_Nagelkerkes_R2'] = nagelkerkes_r2
                                seed_results[f'{scenario}_Tjur_R2'] = tjur_R2
                        
                        # Append the results for the current seed to the results list
                        results.append(seed_results)
                        del temp_df, temp_df_scenario, seed_results

                    # Convert results to a DataFrame
                    results_df = pd.DataFrame(results)

                    # Save to CSV
                    output_csv_path = path_temp + 'COMPILED_' + title + '.csv'
                    results_df.to_csv(output_csv_path, index=False)

                    print(f'Results saved to {output_csv_path}')
                    del results, results_df, df1, title, path_temp, file_name, output_csv_path
