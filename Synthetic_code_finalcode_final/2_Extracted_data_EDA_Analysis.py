# This script is used to extract samples and SNPs from plink file and save it in a csv file.

# import libraries
import pandas as pd
import numpy as np
import gc
# set all random seeds to 0
np.random.seed(0)
# import random and set seed
import random
random.seed(0)
import os
import matplotlib.pyplot as plt
# suppress all warnings
import warnings
warnings.filterwarnings("ignore")
save_path = 'C:/Users/skumar26/synthetic_data_Generation_RESULTS/Gaps_plots/'

def data_eda():
    data_eur = pd.read_csv('data_eur.csv')
    data_eur.iloc[3:, :] = data_eur.iloc[3:, :].astype(float)
    for race in ["amr", "sas", "eas", "afr"]:
        data_ddp = pd.read_csv(f'data_{race}.csv')
        # change datatypes of all columns to float
        data_ddp.iloc[3:, :] = data_ddp.iloc[3:, :].astype(float)
        print(' ')
        print(f'Before processing EUR: {np.unique(data_eur.iloc[3:, :].values, return_counts=True)}')
        print(f'Before Processing {race}: {np.unique(data_ddp.iloc[3:, :].values, return_counts=True)}')        # change datatypes of all columns to float
        
        list_uniq_idx = []
        eur_allele_1= []
        ddp_allele_1 = []
        eur_zeros_freq = []
        eur_two_freq = []
        ddp_zeros_freq = []
        ddp_two_freq = []
        for i in range(data_eur.shape[1]):
            idx = 1
            if data_eur.iloc[idx,i]!=data_ddp.iloc[idx,i]:
                if (data_eur.iloc[idx,i][2]!=data_ddp.iloc[idx,i][7]) or (data_eur.iloc[idx,i][7]!=data_ddp.iloc[idx,i][2]):
                    print('allele wildly different')
                    exit()
                else:
                    list_uniq_idx.append(i)
                    eur_allele_1.append(data_eur.iloc[idx,i][2])
                    ddp_allele_1.append(data_ddp.iloc[idx,i][2])
                    eur_zeros_freq.append((data_eur.iloc[:,i]==0).sum()/(data_eur.shape[0]-2))
                    eur_two_freq.append((data_eur.iloc[:,i]==2).sum()/(data_eur.shape[0]-2))
                    ddp_zeros_freq.append((data_ddp.iloc[:,i]==0).sum()/(data_ddp.shape[0]-2))
                    ddp_two_freq.append((data_ddp.iloc[:,i]==2).sum()/(data_ddp.shape[0]-2))
                    # flip 0 and 2 in the data_ddp column
                    data_ddp.iloc[3:,i] = data_ddp.iloc[3:,i].replace(0, 3)
                    data_ddp.iloc[3:,i] = data_ddp.iloc[3:,i].replace(2, 0)
                    data_ddp.iloc[3:,i] = data_ddp.iloc[3:,i].replace(3, 2)
        print(f'EUR vs {race}: A total of {len(list_uniq_idx)} SNPs, where reversal of allele pair (dominant, recessive) is observed')
        df_EUR_vs_DDP = pd.DataFrame(columns = data_eur.iloc[0, list_uniq_idx].values)
        df_EUR_vs_DDP.loc[0] = [(data_eur.iloc[1, i], data_eur.iloc[2, i]) for i in list_uniq_idx]
        df_EUR_vs_DDP.loc[1] = [(data_ddp.iloc[1, i], data_ddp.iloc[2, i]) for i in list_uniq_idx]
        df_EUR_vs_DDP.loc[2] = ' '
        df_EUR_vs_DDP.loc[3] = eur_allele_1
        df_EUR_vs_DDP.loc[4] = eur_zeros_freq
        df_EUR_vs_DDP.loc[5] = eur_two_freq
        df_EUR_vs_DDP.loc[6] = ' '
        df_EUR_vs_DDP.loc[7] = ddp_allele_1
        df_EUR_vs_DDP.loc[8] = ddp_zeros_freq
        df_EUR_vs_DDP.loc[9] = ddp_two_freq
      
        # set the index of first row to be EUR_allele_pairs
        df_EUR_vs_DDP.index = ['EUR_allele_pairs', f'{race}_allele_pairs', 'EUR', 'Dominant_allele_EUR', 'EUR_zero_freq', 'EUR_two_freq',  race, f'Dominant_allele_{race}', f'{race} zero_freq', f'{race} two_freq']
        # save the dataframe
        df_EUR_vs_DDP.to_csv(save_path+f'EUR_vs_{race}.csv')
        print(f'EUR: {np.unique(data_eur.iloc[3:, :].values, return_counts=True)}')
        print(f'{race}: {np.unique(data_ddp.iloc[3:, :].values, return_counts=True)}')
        # save data_ddp
        data_ddp.to_csv(f'data_{race}_flipped.csv', index = False)
        del data_ddp, df_EUR_vs_DDP, list_uniq_idx, eur_allele_1, ddp_allele_1, eur_zeros_freq, eur_two_freq, ddp_zeros_freq, ddp_two_freq
        gc.collect()

                

# main function
def main():
    data_eda()
if __name__ == "__main__":
    main()

