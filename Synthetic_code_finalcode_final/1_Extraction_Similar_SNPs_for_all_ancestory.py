# This script is used to extract samples and SNPs from plink file and save it in a csv file.

# import libraries
import pandas as pd
import numpy as np
import gc
# set all random seeds to 0
np.random.seed(0)
import random
random.seed(0)
import os
import matplotlib.pyplot as plt
# install pip install bed-reader
from bed_reader import open_bed, sample_file

def common_SNP_func(path, chr_num):
    sid_dict = {}
    for race in ['EAS', 'AFR', 'AMR', 'EUR', 'SAS']:
        file_path = path+'/'+race +'/'
        bed_file = file_path + 'chr'+str(chr_num)+'.bed'   
        bed = open_bed(bed_file)
        temp_sid = bed.sid
        sid_dict[race] = [i.split(':')[0]+':'+i.split(':')[1] for i in temp_sid]
        del temp_sid
        if race == 'EAS':
            common_snps = sid_dict['EAS']
        common_snps = np.intersect1d(common_snps, sid_dict[race])
        del bed, bed_file
        gc.collect()
    return sid_dict, common_snps

def common_Index_f_SNP(sid_dict, common_snps):
    temp_df = pd.DataFrame()
    len_common_snps = len(common_snps)
    for race in ['EAS', 'AFR', 'AMR', 'EUR', 'SAS']:
        common_snps_check, _, Commons2 = np.intersect1d(common_snps, sid_dict[race],
                                                            return_indices=True)
        # check if the length of common snps is same as the length of common snps check
        # print error if not
        if len_common_snps != len(common_snps_check):
            print('...error in common snps..........')
            exit()
        else:
            temp_df[race] = Commons2
        # concatentate the dataframes
        del common_snps_check, Commons2
    return temp_df

def CHR_SNP_RACE_index(path):
    df_common_snps_index = pd.DataFrame()
    for chr_num in range(1,23):
        sid_dict, common_snps = common_SNP_func(path, chr_num)
        if chr_num == 1:
            df_common_snps_index = common_Index_f_SNP(sid_dict, common_snps)
            df_common_snps_index['chr'] = chr_num
            df_common_snps_index['common_snps'] = ['chr' + str(chr_num)+'_'+i for i in common_snps]
        else:
            temp_df = common_Index_f_SNP(sid_dict, common_snps)
            temp_df['chr'] = chr_num
            temp_df['common_snps'] = ['chr' + str(chr_num)+'_'+i for i in common_snps]
            df_common_snps_index = pd.concat([df_common_snps_index, temp_df], axis=0)
            del temp_df
        del sid_dict, common_snps
        gc.collect()      
    return df_common_snps_index

def extract_samples_snps(file_path, race, bed_file, n=None, selected_sample=None, 
                         selected_variant=None, common_snps=None, seed=0):
    # read bed file
    np.random.seed(seed)
    file_path = file_path+'/'+race +'/'
    bed_file = file_path + bed_file + '.bed'    

    bed = open_bed(bed_file)
    data = bed.read()
    allele_1_name = bed.allele_1
    allele_2_name = bed.allele_2
    # extract m SNPs
    if selected_variant is None:
        print('Error: pass selected_variant as parameter')
        # terminate the program
        exit()

    data = data[:, selected_variant]
    allele_1_name = allele_1_name[selected_variant]
    allele_2_name = allele_2_name[selected_variant]
    # merge both arrays as a tuple
    allele_name = list(zip(allele_1_name, allele_2_name))
    # extract n samples    
    if selected_sample is None:
        selected_sample = list(np.random.choice(data.shape[0], n, replace=False))
        print('selected_samples randomly')

    data = data[selected_sample, :]
    # count 0 of each column 
    count_0 = np.count_nonzero(data == 0, axis=0)
    # count 1 of each column
    count_1 = np.count_nonzero(data == 1, axis=0)
    # count 2 of each column
    count_2 = np.count_nonzero(data == 2, axis=0)
    total_count = count_0 + count_1 + count_2

    if (total_count.max() != total_count.min()) or (total_count.max() != len(selected_sample)):
        print('Error: allele_total_count.max() != allele_total_count.min() or allele_total_count.max() != len(selected_sample)')
        # terminate the program
        exit()

    freq_allele_0 = list((2*count_0 + count_1)/(2*total_count))

    data = pd.DataFrame(data)

    temp_df = pd.DataFrame([common_snps], columns=data.columns)
    temp_df.loc[len(temp_df)] = allele_name
    temp_df.loc[len(temp_df)] = freq_allele_0
    data = pd.concat([temp_df, data], axis=0)
    del temp_df
    return data, selected_sample

def data_extraction_ch(path):
    # extract samples and SNPs
    n=100000
    m=3000
    # extract df with chr and range
    df = CHR_SNP_RACE_index(path)
    # save df
    # print length
    print('length of common snps: ', len(df))
    df.to_csv('common_snps_ids.csv', index=False)
    # extract m rows randomly from df
    df_snps = df.sample(n=m, axis=0, random_state=0)
    # reset index
    df_snps.reset_index(drop=True, inplace=True)
    del df
    gc.collect()
    print('group by chr column and count the number of rows', df_snps.groupby('chr').count().iloc[:,0])
    data_dict = {}
    selected_samples_dict = {}
    for sh_i in range(1, 23):
        temp_SNP_idx = df_snps[df_snps['chr']==sh_i]
        temp_SNP_idx.reset_index(drop=True, inplace=True)
        for race in ['EAS', 'EUR', 'AFR', 'AMR', 'SAS']:
            if sh_i == 1:
                data_dict[race], selected_samples_dict[race] = extract_samples_snps(path, race, bed_file='chr'+str(sh_i), n=n, seed=sh_i, 
                                                                                                        selected_variant=temp_SNP_idx[race].values, 
                                                                                                        common_snps=temp_SNP_idx['common_snps'].values)
                
                gc.collect()
            else:
                # extract samples and SNPs for previous selection of samples
                data_temp, _                                = extract_samples_snps(path, race, bed_file='chr'+str(sh_i), seed=sh_i, 
                                                                                            selected_variant=temp_SNP_idx[race].values,
                                                                                            selected_sample=selected_samples_dict[race],
                                                                                            common_snps=temp_SNP_idx['common_snps'].values)
                
                # concatenate dataframes
                data_dict[race]= pd.concat([data_dict[race], data_temp], axis=1)
                del data_temp
                gc.collect()
        del temp_SNP_idx
        gc.collect()
        print(' all data extracted for chr'+str(sh_i))

    for race in ['EAS', 'EUR', 'AFR', 'AMR', 'SAS']:
        print(f'{race} data shape {data_dict[race].shape}')
        data_dict[race].to_csv(f'data_{race.lower()}.csv', index=False)
    print('.......data extraction completed........')

# main function
def main():
    # extract samples and SNPs

    path = 'C:/Users/skumar26/dataverse_files/GenotypeData'
    data_extraction_ch(path)
if __name__ == "__main__":
    main()

         

