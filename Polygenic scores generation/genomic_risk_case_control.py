# Purpose: Generate case and control data for euro and ddp SNP data.
'''
Created on 2023-06-09
Author: skumar26
Version 1.0

'''

import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

def calculate_g(w: np.ndarray, x: np.ndarray, h_sq: float) -> np.ndarray:
    z = np.random.normal(0, np.sqrt(1 - h_sq), x.shape[0]).reshape(-1, 1)            
    dot_vector = np.dot(x, w).reshape(-1, 1)

    dot_vector = np.sqrt(h_sq) * dot_vector/np.std(dot_vector)
    g = dot_vector + z
    print(f'h_sq : {h_sq}, heritability calculated: {np.var(dot_vector) / np.var(g)}')
    assert g.shape[0] == dot_vector.shape[0] == x.shape[0]
    corr_g = np.corrcoef(g.T, dot_vector.T)[0, 1]
    print(f'corr(g, dot_vector) : {corr_g}')
    return g, corr_g

# def calculate_weights_ddp(w: np.ndarray, rho: float) -> np.ndarray:
#     z = np.random.normal(0, 1, w.shape[0])
#     w_ddp = rho * w + np.sqrt(1 - rho ** 2) * z
#     return w_ddp

# def calculate_weights_ddp(eur_w: np.ndarray, rho: float) -> np.ndarray:
#     x= np.random.normal(0,1,eur_w.shape[0])
#     # calculte residuals of the least squares regression of x on eur_w
#     x_res = x - np.dot(np.dot(x,eur_w)/np.dot(eur_w,eur_w),eur_w)
#     correlated_x = rho * np.std(x_res) * eur_w + np.sqrt(1 - rho**2) * np.std(eur_w) * x_res
#     return correlated_x

def generate_case_control_data(h_sq: float, rho: float, eur_data: pd.DataFrame, ddp_data: pd.DataFrame, random_seed: int, race_seed: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Generates case and control data for euro and ddp SNP data.

    Args:
        h_sq: Heritability.
        rho: Correlation factor.
        data_eur: Euro SNP data.
        data_ddp: DDP SNP data.

    Returns:
        Euro and ddp SNP data with case and control data.
    """  
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
   
    corr_matrix= pd.DataFrame([[1, rho, rho, rho, rho], [rho, 1, rho, rho, rho], [rho, rho, 1, rho, rho], [rho, rho, rho, 1, rho], [rho, rho, rho, rho, 1]], columns=['w_eur', 'w_ddp', 'w_amr', 'w_sas', 'w_eas'], index=['w_eur', 'w_ddp', 'w_amr', 'w_sas', 'w_eas'])
    print(corr_matrix)
    w_eur, w_amr, w_sas, w_eas, w_afr = np.random.multivariate_normal([0,0,0,0,0], corr_matrix.values, eur_data.shape[1]).T

    if race_seed == 101:
        w_ddp = w_amr
        print (' For w_ddp = w_amr')
    elif race_seed == 202:
        w_ddp = w_sas
        print (' For w_ddp = w_sas')
    elif race_seed == 303:
        w_ddp = w_eas
        print (' For w_ddp = w_eas')
    elif race_seed == 404:
        w_ddp = w_afr
        print (' For w_ddp = w_afr')
    # assert length of weight vector is equal to number of columns in euro data and ddp data
    assert w_eur.shape[0] == eur_data.shape[1]
    assert w_ddp.shape[0] == ddp_data.shape[1]
    #  find correlation between euro and ddp weight vector       
    corr = np.corrcoef(w_eur, w_ddp)[0, 1]
    print(f'rho : {rho} and, corr(eur_w, ddp_w) : {corr}')
    
    # euro genomic risk
    g_eur, corr_g_eur = calculate_g(w_eur, eur_data.values, h_sq)

    # ddp genomic risk
    g_ddp, corr_g_ddp = calculate_g(w_ddp, ddp_data.values, h_sq)
    
    while np.abs(corr_g_eur - corr_g_ddp) > 0.008:
        g_ddp, corr_g_ddp = calculate_g(w_ddp, ddp_data.values, h_sq)

    # 50% of the data is case and 50% of the data is control
    g_eur_50 = np.percentile(g_eur, 50)
    eur_data['case'] = np.where(g_eur > g_eur_50, 1, 0).reshape(-1, 1)
    g_ddp_50 = np.percentile(g_ddp, 50)
    ddp_data['case'] = np.where(g_ddp > g_ddp_50, 1, 0).reshape(-1, 1)

    return eur_data, ddp_data    # return case control data for euro and ddp data

# main function
def main():   
    # read command line arguments
    h_sq = float(sys.argv[1])
    rho = float(sys.argv[2])
   
    # read euro data and ddp data from file
    race_ddp='AMR'
    data_eur = pd.read_csv('data_eur.csv').iloc[3:,:].astype(float)
    data_ddp = pd.read_csv('data_'+race_ddp+'_flipped.csv').iloc[3:,:].astype(float)

    # generate case and control data for euro and ddp data
    data_eur, data_ddp = generate_case_control_data(h_sq, rho, data_eur, data_ddp, 0, 101)

    # print unique values and counts of euro data without case control
    print(np.unique(data_eur[data_eur.columns[:-1]].values, return_counts=True))
    # print unique values and counts of ddp data without case control
    print(np.unique(data_ddp[data_ddp.columns[:-1]].values, return_counts=True))

    # print euro case control unique values and counts
    print(data_eur['case'].value_counts())
    # print ddp case control unique values and counts
    print(data_ddp['case'].value_counts())
    # print euro data shape
    print(data_eur.shape)
    #  print ddp data shape
    print(data_ddp.shape)

    # print euro data head 2 rows
    print(data_eur.head(2))
    # print ddp data head 2 rows
    print(data_ddp.head(2)) 

# call main function
if __name__ == "__main__":
    main()
    
# command line
#  C:/Users/skumar26/AppData/Local/Programs/Python/Python311/python.exe genomic_risk_case_control.py 0.5 0.8 
# py genomic_risk_case_control.py 0.5 0.9 




