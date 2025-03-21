# pip install -U kaleido

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import os
import gc
import math

h_sq_list = [0.25, 0.5]
rho_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
lambda__list = [0.1, 0.2, 0.3, 0.4, 0.5]
dict_gap = {}
metric='_sq_corr'
dict_gap[f'Mix1{metric}']= [f'Mix2{metric}']
dict_gap[f'ind_1{metric}']= [f'ind_2{metric}']
# dict_gap['tl2{metric}']= [f'Mix2{metric}', f'ind_2{metric}']
dict_gap['tl2_Vs_Mix2_'] = [f'Mix2{metric}']
dict_gap['tl2_Vs_ind_2_'] = [f'ind_2{metric}']
# dict_gap[f'tl2_Mixed{metric}'] = [f'tl2{metric}']
size_max = 0
size_min = 100
# make a new directory to save the plots pf gaps
os.makedirs('C:/Users/skumar26/synthetic_data_Generation_RESULTS/Gaps_plots', exist_ok=True)
 
for temp_file_prefix in [
    # 'xgboost_latest_regression',
    'neural_regression_random_genomic_val_test_together_40_runs_3000SNPs_100000_patients'
    # 'neural_10000P_100SNPs_regression',
    # 'XGBoost_10000P__100SNPs_regression',
    # 'tabtransformer_regression',
    # 'neural_classification_10000P_500SNPs'
    ]:
    print(temp_file_prefix)
    for h_sq in h_sq_list:
        fig_col = 5
        fig_row = len(dict_gap.keys())
        fig_heat, axs = plt.subplots(fig_row, fig_col, figsize=(20, 10))
        heat_col= 0
        for ddp_str in ['eur', 'amr', 'sas', 'eas', 'afr']:
            df_3d = pd.DataFrame(columns=['ddp_str', 'rho', 'lambda', 'key', 'median_gap', 'std_gap', 'pvalue', metric])  
            row_idx = 0
            for key in dict_gap.keys():
                for rho in rho_list:
                    for lambda_ in lambda__list:
                        file_name = 'h_sq_' + str(h_sq) + '_rho_' + str(rho) + '_lambda_' + str(lambda_) + ddp_str
                        file_name =  'C:/Users/skumar26/synthetic_data_Generation_RESULTS/'+ temp_file_prefix+f'/{ddp_str}/COMPILED_' + file_name + '.csv'
                        # read file if exists
                        if not os.path.exists(file_name):
                            print('Error: file does not exist: ', file_name)
                            df_3d.loc[row_idx] = [ddp_str, rho, lambda_, key[:-1].upper(), -0, -0, 0,
                                                '('+'NA'+'/' + 'NA'+')']
                            row_idx += 1
                        else:
                            df = pd.read_csv(file_name)
                            # replace any negative value with 0 in whole dataframe

                            if key == 'tl2_Vs_Mix2_' or key == 'tl2_Vs_ind_2_':
                                temp_series = df[f'tl2{metric}'] - df[dict_gap[key]].median(axis=1)
                                # check if these columns are negative in df[dict_gap[key]]
                                if (df[f'tl2{metric}'] < 0).any():
                                    print(f'Error: negative value in the dataframe: {f"tl2{metric}"}')
                            else:
                                temp_series = df[key] - df[dict_gap[key]].median(axis=1)
                                # check if these columns are negative
                                if (df[key] < 0).any():
                                    print(f'Error: negative value in the dataframe: {key}')
                            if (df[dict_gap[key]]<0).any().any():
                                print(f'Error: negative value in the dataframe: {dict_gap[key]}')
                            
                            if len(temp_series) != 40:
                                print('Error: length of temp_series is not 40')
                                exit()
                            # pvalue one tailed test
                            if key == 'tl2_Vs_Mix2_' or key == 'tl2_Vs_ind_2_':
                                pvalue = -1*math.log10(stats.mannwhitneyu(df[f'tl2{metric}'], df[dict_gap[key]].median(axis=1), alternative='greater')[1])
                                df_3d.loc[row_idx] = [ddp_str, rho, lambda_, key[:-1].upper(), temp_series.median(), temp_series.std(), pvalue,
                                                    '('+str(int((df[f'tl2{metric}'].median()*100).round(0)))+'/' + str(int((df[dict_gap[key]].median(axis=1).median()*100).round(0)))+')']
                                
                                
                                                    
                            else:
                                pvalue = -1*math.log10(stats.mannwhitneyu(df[key], df[dict_gap[key]].median(axis=1), alternative='greater')[1])
                                df_3d.loc[row_idx] = [ddp_str, rho, lambda_, key[:-1].upper(), temp_series.median(), temp_series.std(), pvalue,
                                                    '('+str(int((df[key].median()*100).round(0)))+'/' + str(int((df[dict_gap[key]].median(axis=1).median()*100).round(0)))+')']
                            # find max and min size of the gap
                            del df
                            if temp_series.median() > size_max:
                                size_max = temp_series.median()
                            if temp_series.median() < size_min:
                                size_min = temp_series.median()
                            # delete the data
                            del temp_series, pvalue
                            gc.collect()
                            row_idx += 1
            # plot heat map for each key, with values as gap and color as pvalue, where all hsq subplots in one figure
            for i, key in enumerate(dict_gap.keys()):
                df_temp = df_3d[df_3d['key']==key[:-1].upper()]
                df_temp_gap = df_temp.pivot(index='rho', columns='lambda', values='median_gap')
                df_temp_pvalue = df_temp.pivot(index='rho', columns='lambda', values='pvalue')
                df_temp_auc = df_temp.pivot(index='rho', columns='lambda', values=metric)
                # create bar light color for min and dark color for max
                cmap = sns.light_palette("green", as_cmap=True)
                # plot heatmap
                if i==fig_row-1 and heat_col==fig_col-1:
                    sns.heatmap(df_temp_pvalue, ax=axs[i, heat_col], annot=df_temp_gap, annot_kws={'va':'bottom'}, fmt='.3f', 
                                vmax=3, vmin=1, cbar=True, cmap=cmap)

                else:
                    sns.heatmap(df_temp_pvalue, ax=axs[i, heat_col], annot=df_temp_gap, annot_kws={'va':'bottom'}, fmt='.3f', 
                                vmax=3, vmin=1, cbar=False, cmap=cmap)
                sns.heatmap(df_temp_pvalue, ax=axs[i, heat_col], annot=df_temp_auc, annot_kws={'va':'top'}, fmt='', 
                            vmax=3, vmin=1, cbar=False, cmap=cmap)            

                # set lambda as x axis label for the last row
                axs[i, heat_col].invert_yaxis()
                if i == fig_row-1:
                    axs[i, heat_col].set_xlabel('lambda')
                else:
                    axs[i, heat_col].set_xlabel('')
                # set title for each subplot on the first column
                if i == 0:
                    axs[i, heat_col].set_title('EUR/'+str(ddp_str))
                else:
                    axs[i, heat_col].set_title('')
                del df_temp, df_temp_gap, df_temp_pvalue
                gc.collect()    
                # when heat_col is 0, then add extra axis to the left of the original one
                if heat_col == 0:
                    # replace Mix_1 with Mixture_Gap, ind_1 with Independent_Gap, tl2 with Transfer_Learning_Gap
                    if key == f'Mix1{metric}':
                        key = 'Mix_Gap'
                    elif key == f'ind_1{metric}':
                        key = 'Ind_Gap'
                    elif key == f'tl2{metric}':
                        key = 'TL_Gap'
                    elif key == 'tl2_Vs_Mix2_':
                        key = 'TL_Vs_Mix2_Gap'
                    elif key == 'tl2_Vs_ind_2_':
                        key = 'TL_Vs_Ind2_Gap'
                    # add extra axis to the left of the original one
                    ax2 = axs[i, heat_col].twinx()
                    # move extra axis to the left, with offset
                    ax2.yaxis.set_label_position('left')
                    ax2.spines['left'].set_position(('axes', -0.2))
                    # hide spine and ticks, set group label
                    ax2.spines['left'].set_visible(False)
                    ax2.set_yticks([])
                    ax2.set_ylabel(key, rotation=90, size='large',
                                ha='right', va='center')
            del df_3d
            gc.collect()
            heat_col += 1   
        # overall title for the figure
        if metric == '':
            if 'regression' not in temp_file_prefix:
                fig_heat.suptitle('h_sq_'+str(h_sq)+'_AUROC', fontsize=16)
            else:
                fig_heat.suptitle('h_sq_'+str(h_sq)+'_R2', fontsize=16)
        else:
            fig_heat.suptitle('h_sq_'+str(h_sq)+metric, fontsize=16)
        # save the figure
        fig_heat.savefig('C:/Users/skumar26/synthetic_data_Generation_RESULTS/Gaps_plots/Heatmap_'+str(h_sq)+ temp_file_prefix+metric+'_Median.png')
        plt.close('all')
        del fig_heat, axs, heat_col
        gc.collect()
    print('size_max: ', size_max)
    print('size_min: ', size_min)





