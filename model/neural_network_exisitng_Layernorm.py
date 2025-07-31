
REGRESSION_FLAG = True
if REGRESSION_FLAG:
    from representation-bias-distribution-shift/Polygenic scores generation/genomic_risk_case_control_regression import *
else:
    from representation-bias-distribution-shift/Polygenic scores generation/genomic_risk_case_control import *

from numpy.random import seed
import numpy as np
import pandas as pd
import random as rn
import os
import gc
import sys
import collections
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import timeit

from tensorflow import keras
from keras import backend as K
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, LayerNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l1_l2
from keras.utils import plot_model

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)

seed(11111)
# set_random_seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)
np.random.seed(11111)
# suppress warning messages
import warnings
warnings.filterwarnings("ignore")
##########################################################################################################
# define a fuction to concatenate euro and ddp data
# and save the data in a new csv file

def concat_euro_ddp(file_path, eur_data, ddp_data, ddp_str, lambda_=0.5):
    # lambda denotes proportion of DDP data compared to EUR data
    # create a new column by adding 'EUR' to the case column
    # reset indexes
    eur_data.reset_index(drop=True, inplace=True)
    ddp_data.reset_index(drop=True, inplace=True)
    # eur cols except case
    eur_cols = eur_data.columns[eur_data.columns != 'case']
    ddp_cols = ddp_data.columns[ddp_data.columns != 'case']
    # assert eur_cols and ddp_cols are equal
    assert len(eur_cols) == len(ddp_cols)
    for i in range(len(eur_cols)):
        assert eur_cols[i] == ddp_cols[i]
    # assert eur_data with eur_cols is not equal to ddp_data with eur_cols
    if ddp_str != 'eur':
        assert (eur_data[eur_cols] != ddp_data[eur_cols]).any().any()
        assert (eur_data != ddp_data).any().any()
        assert (eur_data['case'] != ddp_data['case']).any()
    del eur_cols, ddp_cols

    eur_data['YR'] =  eur_data['case'].astype(str) + 'EUR'
    # create a new column R and assign 'EUR' to all the rows
    eur_data['R'] = 'EUR'
    # read the DDP csv file
    # create a new column by adding 'DDP' to the case column
    ddp_data['YR'] = ddp_data['case'].astype(str)+ ddp_str
    # create a new column R and assign 'DDP' to all the rows
    ddp_data['R'] = ddp_str
    # find the length of the concatenated data
    # len_ddp_data = len(pd.concat([eur_data, ddp_data], axis=0))
    len_ddp_data = len(ddp_data)
    assert len(eur_data) == len_ddp_data
    if REGRESSION_FLAG:
        data = pd.concat([eur_data.sample(int(len_ddp_data*(1-lambda_)), random_state=1),
                            ddp_data.sample(int(len_ddp_data*lambda_), random_state=1)], axis=0)
    else:
        data = pd.concat([eur_data.groupby('YR', group_keys=False).apply(lambda x: x.sample(frac= ((1-lambda_)*len_ddp_data)/len(eur_data), replace=False, random_state = 1)),
                              ddp_data.groupby('YR', group_keys=False).apply(lambda x: x.sample(frac= (lambda_*len_ddp_data)/len(ddp_data), replace=False, random_state = 1))], axis=0)
            # print unique counts of YR
        print('Unique counts of YR: ', collections.Counter(data['YR'].values))

    # percentage of euro data and ddp data in the concatenated data
    print('Percentage of EUR data: ', len(data[data['R']=='EUR'])/len(data)*100, '%', 
          'Percentage of DDP data: ', len(data[data['R']==ddp_str])/len(data)*100, '%')
    print('Total data shape: ', data.shape)

    assert len(data['R'].unique()) == 2
    # print len of unique values of case and R columns
    print(f'Unique counts of Case column {len(data["case"].unique())} and R column {collections.Counter(data["R"].values)}')
    # assert that number of unique values in case column is minimum of 3
    if REGRESSION_FLAG:
        assert len(data['case'].unique()) >= 3
    else:
        assert len(data['case'].unique()) == 2
    
    # rename case column to Y
    data.rename(columns={'case': 'Y'}, inplace=True)
    # save the data in a new csv file
    return data
##########################################################################################################
def get_data_for_all_models(data, ddp_str, seed):
    # standardize the the features except Y, R, YR columns
    scaler = StandardScaler()
    df = data.copy()
    df[df.columns[:-3]] = scaler.fit_transform(df[df.columns[:-3]])
    # split the data into train, validation and test data
    if REGRESSION_FLAG:
        train, test = train_test_split(df, test_size=0.25, random_state=seed, shuffle=True, stratify=df['R'])
    else:
        train, test = train_test_split(df, test_size=0.25, random_state=seed, shuffle=True, stratify=df['YR'])
    print('Train data shape: ', train.shape, 'Test data shape: ', test.shape)
    # split the train into train data and its labels
    Y_train, R_train = train['Y'].values.astype('float32'), train['R'].values

    if REGRESSION_FLAG:
        print('Unique counts of R columns in train data: ', collections.Counter(train['R'].values))
    else:
        # print unique count of values of Y, R, YR columns in train data
        print('Unique counts of Y, R, YR columns in train data: ', collections.Counter(train['Y'].values), 
              collections.Counter(train['R'].values), collections.Counter(train['YR'].values))

    train = train.drop(columns=['Y', 'R', 'YR'])
    X_train = train.values

    # split the test into validation and test data
    # if REGRESSION_FLAG:
    #     val_samples, test_samples = train_test_split(test, test_size=0.5, random_state=seed, shuffle=True, stratify=test['R'])
    # else:
    #     val_samples, test_samples = train_test_split(test, test_size=0.5, random_state=seed, shuffle=True, stratify=test['YR'])
    val_samples = test.copy()
    test_samples = test.copy()
    if REGRESSION_FLAG:
        print('Unique counts of R columns in val data: ', collections.Counter(val_samples['R'].values))
    else:
        # print unique count of values of Y, R, YR columns in test data
        print('Unique counts of Y, R, YR columns in test data: ', collections.Counter(test['Y'].values),
              collections.Counter(test['R'].values), collections.Counter(test['YR'].values))
    
    # split the validation and test data into data and its labels
    Y_val, R_val = val_samples['Y'].values.astype('float32'), val_samples['R'].values
    # split the test data into data and its labels
    Y_test, R_test = test_samples['Y'].values.astype('float32'), test_samples['R'].values
    # drop the Y, R, YR columns from the data
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_val, X_test = val_samples.values, test_samples.values
    # prepare the data for all models
    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test, R_test)
    # prepare the data for EA 
    EA_X_train, EA_Y_train = X_train[R_train == 'EUR'], Y_train[R_train == 'EUR']
    EA_X_val, EA_Y_val = X_val[R_val == 'EUR'], Y_val[R_val == 'EUR']
    EA_X_test, EA_Y_test = X_test[R_test == 'EUR'], Y_test[R_test == 'EUR']
    # prepare the data for EAA
    EAA_X_train, EAA_Y_train = X_train[R_train == ddp_str], Y_train[R_train == ddp_str]
    EAA_X_val, EAA_Y_val = X_val[R_val == ddp_str], Y_val[R_val == ddp_str]
    EAA_X_test, EAA_Y_test = X_test[R_test == ddp_str], Y_test[R_test == ddp_str]

    # prepare the EA data for all models
    EA_train_data = (EA_X_train, EA_Y_train)
    EA_val_data = (EA_X_val, EA_Y_val)
    EA_test_data = (EA_X_test, EA_Y_test)
    # prepare the EAA data for all models
    EAA_train_data = (EAA_X_train, EAA_Y_train)
    EAA_val_data = (EAA_X_val, EAA_Y_val)
    EAA_test_data = (EAA_X_test, EAA_Y_test)

    # return the data
    Aggr = [train_data, val_data, test_data]
    EA = [EA_train_data, EA_val_data, EA_test_data]
    EAA = [EAA_train_data, EAA_val_data, EAA_test_data]
    print('data prepared for all models successfully completed')
    return [Aggr, EA, EAA]
##########################################################################################################

def history_plot(history, title_fig):
    plt.rcParams.update({'font.size': 16})
    # hist = pd.DataFrame(history.history)
    # fig, (ax1) = plt.subplots(figsize=(12,12),nrows=1, ncols=1)
    # hist['auc'].plot(ax=ax1,c='k',label='training AUC')
    # hist['val_auc'].plot(ax=ax1,c='r',linestyle='--', label='validation AUC')
    # ax1.legend()
    # plt.ylabel('AUC',size=14)
    # plt.xlabel('Epoch',size=14)
    # plt.title(title_fig)
    # plt.savefig(title_fig+'.png', dpi=300)

##########################################################################################################
# hidden_layers_sizes: a list of hidden layer sizes, e.g. [10, 10] means two hidden layers with 10 neurons each
def build_model(n_in,
             learning_rate=0.001, hidden_layers_sizes=None,
             lr_decay=0.0, momentum=0.9,
             L2_reg=0.0, L1_reg=0.0,
             activation="relu",
             dropout=None,
             input_dropout=None):
    # build model 
    model_input = Input(shape=(n_in,), dtype='float32')
    
    if hidden_layers_sizes == None:
        z = Dense(1, name='OutputLayer')(model_input)
    else:
        for idx in range(len(hidden_layers_sizes)):
            layer_size = hidden_layers_sizes[idx]
            if idx == 0:
                if input_dropout:
                    input = Dropout(input_dropout)(model_input)
                else:
                    input = model_input
                z = Dense(layer_size, activation=activation, name='HiddenLayer',
                          kernel_initializer=keras.initializers.glorot_uniform(seed=11111),
                          kernel_regularizer=l1_l2(l1=L1_reg, l2=L2_reg), bias_regularizer=l1_l2(l1=L1_reg, l2=L2_reg))(
                    input)
                z = LayerNormalization()(z)
            else:
                z = Dense(layer_size, activation=activation, name='HiddenLayer' + str(idx + 1),
                      kernel_initializer=keras.initializers.glorot_uniform(seed=11111),
                      kernel_regularizer=l1_l2(l1=L1_reg, l2=L2_reg), bias_regularizer=l1_l2(l1=L1_reg, l2=L2_reg))(z)
                z = LayerNormalization()(z)
            if dropout:
                z = Dropout(rate=dropout)(z)
        z = Dense(1, name='OutputLayer')(z)
    if REGRESSION_FLAG:
        model_output = Activation('linear')(z)
    else:
        model_output = Activation('sigmoid')(z)
    model = Model(model_input, model_output)

    # compile model
    optimizer = Adam(learning_rate=learning_rate)

    if REGRESSION_FLAG:
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])
    else:
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[keras.metrics.AUC(name='auc')])
    model.summary()

    return model
##########################################################################################################
def independent_learning(data, i_str, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    train_fig_paths = kwds.pop('train_fig_paths')
    train_data, val_data, test_data = data
    X_test, Y_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    if REGRESSION_FLAG:
        es = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1, patience=30, restore_best_weights=True)
    else:
        es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=30, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3,
                               patience = 4, min_lr=0.000001, 
                               mode = 'min', verbose = 1)
    history = model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es, reduce], verbose=0)
    history_plot(history, title_fig = train_fig_paths +' Independent learning '+i_str)

    x_test_scr = np.round(model.predict(X_test), decimals=3)
    # save the model with train_fig_paths name
    # model.save(train_fig_paths + '_Independent_learning_'+i_str+'_model.h5')
    if REGRESSION_FLAG:
        R2_score = r2_score(Y_test, x_test_scr)
        print(f'Independent learning R2 {i_str} = {R2_score}')
        del history
        return R2_score, x_test_scr, model
    else:
        AUC = roc_auc_score(Y_test, x_test_scr)
        # calculate PR AUC score
        precision_v, recall_v, _ = precision_recall_curve(Y_test, x_test_scr)
        pr_auc_score = auc(recall_v, precision_v)
        print(f'Independent learning AUC {i_str} = {AUC}, PR AUC = {pr_auc_score}')
        del history, precision_v, recall_v
        return AUC, pr_auc_score, x_test_scr, model
################################################################
def mixture_learning(Aggr, ddp_str='DDP', **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    train_fig_paths = kwds.pop('train_fig_paths')
    train_data, val_data, test_data = Aggr
    print('Mixture learning train_data', train_data[0].shape, train_data[1].shape)
    print('Mixture learning val_data', val_data[0].shape, val_data[1].shape)
    print('Mixture learning test_data', test_data[0].shape, test_data[1].shape)

    X_test, Y_test, R_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    if REGRESSION_FLAG:
        es = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1, patience=30, restore_best_weights=True)
    else:
        es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=30, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3,
                               patience = 4, min_lr=0.000001, 
                               mode = 'min', verbose = 1)    
    history = model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es, reduce], verbose=0)
    history_plot(history, title_fig = train_fig_paths + ' Mixture learning')
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    if REGRESSION_FLAG:
        R2_score = r2_score(Y_test, x_test_scr)
        print(f'Mixture learning R2: = {R2_score}')
    else:
        A_AUC = roc_auc_score(Y_test, x_test_scr)
        # calculate PR AUC score
        precision_v, recall_v, _ = precision_recall_curve(Y_test, x_test_scr)
        pr_auc_score = auc(recall_v, precision_v)
    temp_val_df = pd.DataFrame(columns= ['col_'+str(i) for i in range(len(Y_test))])
    # add y_test, x_test_scr and R_test as rows to temp_val_df
    temp_val_df.loc[len(temp_val_df)] = Y_test
    temp_val_df.loc[len(temp_val_df)] = x_test_scr.reshape(-1)
    temp_val_df.loc[len(temp_val_df)] = [1 if i == 'EUR' else 0 for i in R_test]
    temp_val_df['info'] = ['Y_test', 'Mix0', 'R_test']
    Y_EA, scr_EA = Y_test[R_test == 'EUR'], x_test_scr[R_test == 'EUR']
    Y_minor, scr_minor = Y_test[R_test == ddp_str], x_test_scr[R_test == ddp_str]
    if REGRESSION_FLAG:
        EA_R2, DDP_R2 = r2_score(Y_EA, scr_EA), r2_score(Y_minor, scr_minor)
        print(f'Mixture learning R2 EUR = {EA_R2}, R2 {ddp_str} = {DDP_R2}')
        Map = {}
        Map["Mix0"], Map["Mix1"], Map["Mix2"] = R2_score, EA_R2, DDP_R2
    else:
        EA_AUC, DDP_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_minor, scr_minor)

        precision_EA, recall_EA, _ = precision_recall_curve(Y_EA, scr_EA)
        precision_minor, recall_minor, _ = precision_recall_curve(Y_minor, scr_minor)
        EA_PR_AUC, DDP_PR_AUC = auc(recall_EA, precision_EA), auc(recall_minor, precision_minor)
        Map = {}
        Map["Mix0"], Map["Mix1"], Map["Mix2"] = A_AUC, EA_AUC, DDP_AUC
        Map["Mix0_PR"], Map["Mix1_PR"], Map["Mix2_PR"] = pr_auc_score, EA_PR_AUC, DDP_PR_AUC
        print(f'Mixture learning AUC {ddp_str} = {A_AUC}, PR AUC = {pr_auc_score}')
    # save the model with train_fig_paths name
    # model.save(train_fig_paths + '_Mixture_learning_model.h5')

    del model
    gc.collect()
    K.clear_session()
    return Map, temp_val_df
################################################################
def naive_transfer(model, EA, Minor, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    train_fig_paths = kwds.pop('train_fig_paths')
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test
    Minor_test_scr = np.round(model.predict(Minor_test_X), decimals=3)
    # save the model with train_fig_paths name
    # model.save(train_fig_paths + '_Naive_transfer_model.h5')
    if REGRESSION_FLAG:
        R2_score = r2_score(Minor_test_Y, Minor_test_scr)
        print(f'Naive transfer learning R2 = {R2_score}')
        del model
        return R2_score, Minor_test_scr
    else:
        Naive_AUC = roc_auc_score(Minor_test_Y, Minor_test_scr)
        # calculate PR AUC score
        precision_v, recall_v, _ = precision_recall_curve(Minor_test_Y, Minor_test_scr)
        pr_auc_score = auc(recall_v, precision_v)
        print(f'Naive transfer learning AUC = {Naive_AUC}, PR AUC = {pr_auc_score}')
        del model
        return Naive_AUC, pr_auc_score, Minor_test_scr
################################################################
def super_transfer(model, EA, Minor, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    train_fig_paths = kwds.pop('train_fig_paths')
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test
    if REGRESSION_FLAG:
        es = EarlyStopping(monitor='val_root_mean_squared_error', mode='min', verbose=1, patience=30, restore_best_weights=True)
    else:
        es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=30, restore_best_weights=True)
    optimizer = Adam(learning_rate=0.0001)
    if REGRESSION_FLAG:
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])
    else:
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[keras.metrics.AUC(name='auc')])
    reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3,
                               patience = 4, min_lr=0.000001, 
                               mode = 'min', verbose = 1)
    history = model.fit(Minor_train_data[0], Minor_train_data[1], validation_data=Minor_val_data, batch_size=batch_size,
              epochs=n_epochs, callbacks=[es, reduce], verbose=0)
    
    history_plot(history, title_fig = train_fig_paths + ' Transfer learning')
    prob = np.round(model.predict(Minor_test_X), decimals=3)
    if REGRESSION_FLAG:
        R2_score = r2_score(Minor_test_Y, prob)
        print(f'Super transfer learning R2 = {R2_score}')
    else:
        A_AUC = roc_auc_score(Minor_test_Y, prob)
        # calculate PR AUC score
        precision_v, recall_v, _ = precision_recall_curve(Minor_test_Y, prob)
        pr_auc_score = auc(recall_v, precision_v)
        print(f'Super transfer learning AUC = {A_AUC}, PR AUC = {pr_auc_score}')
    # save the model with train_fig_paths name
    # model.save(train_fig_paths + '_Super_transfer_model.h5')
    del model
    gc.collect()
    K.clear_session()
    if REGRESSION_FLAG:
        return R2_score, prob
    else:
        return A_AUC, pr_auc_score, prob
################################################################
def run_cv(data,ddp_str, seed, train_fig_paths):
    epoch = 500
    para_mix = {'batch_size': 32, 'n_epochs': epoch, 'dropout': 0.1, 'L1_reg': 0.001, 'L2_reg': 0.001, 
                'hidden_layers_sizes': [200], "train_fig_paths": train_fig_paths}

    Aggr, EA, Minor = data

    Map = {}
    Map, temp_df_pred = mixture_learning(Aggr,ddp_str=ddp_str, **para_mix)
    gc.collect()
    K.clear_session()
    if REGRESSION_FLAG:
        R2_temp, eur_IND, eur_model = independent_learning(EA, i_str = 'EUR' ,**para_mix)
        Map['ind_1'] = R2_temp
    else:
        auc_temp, pr_auc_temp, eur_IND, eur_model = independent_learning(EA, i_str = 'EUR' ,**para_mix)
        Map['ind_1'] = auc_temp
        Map['ind_1_PR'] = pr_auc_temp
    # convert eur_IND to a list make it equal to the temp_df_pred.shape[1]
    eur_IND = eur_IND.reshape(-1).tolist()
    eur_IND = eur_IND + [None]*(temp_df_pred.shape[1] - len(eur_IND)-1)+ ['eur_IND']
    temp_df_pred.loc[len(temp_df_pred)] = eur_IND
    if REGRESSION_FLAG:
        del R2_temp, eur_IND
    else:
        del auc_temp, pr_auc_temp, eur_IND
    gc.collect()
    K.clear_session()
    if REGRESSION_FLAG:
        R2_temp, ddp_IND, _ = independent_learning(Minor, i_str = ddp_str, **para_mix)
        Map['ind_2'] = R2_temp
    else:
        auc_temp, pr_auc_temp, ddp_IND, _ = independent_learning(Minor, i_str = ddp_str, **para_mix)
        Map['ind_2'] = auc_temp
        Map['ind_2_PR'] = pr_auc_temp
    # convert ddp_IND to a list make it equal to the temp_df_pred.shape[1]
    ddp_IND = ddp_IND.reshape(-1).tolist()
    ddp_IND = ddp_IND + [None]*(temp_df_pred.shape[1] - len(ddp_IND)-1)+ [ddp_str+'_IND']
    temp_df_pred.loc[len(temp_df_pred)] = ddp_IND
    if REGRESSION_FLAG:
        del R2_temp, ddp_IND
    else:
        del auc_temp, pr_auc_temp, ddp_IND
    gc.collect()
    K.clear_session()
    if REGRESSION_FLAG:
        R2_temp, naive_scr = naive_transfer(eur_model, EA, Minor, **para_mix)
        Map['naive'] = R2_temp
    else:
        auc_temp, pr_auc_temp, naive_scr = naive_transfer(eur_model, EA, Minor, **para_mix)
        Map['naive'] = auc_temp
        Map['naive_PR'] = pr_auc_temp
    # convert naive_scr to a list make it equal to the temp_df_pred.shape[1]
    naive_scr = naive_scr.reshape(-1).tolist()
    naive_scr = naive_scr + [None]*(temp_df_pred.shape[1] - len(naive_scr)-1)+ ['naive']
    temp_df_pred.loc[len(temp_df_pred)] = naive_scr
    if REGRESSION_FLAG:
        del R2_temp, naive_scr
    else:
        del auc_temp, pr_auc_temp, naive_scr
    gc.collect()
    K.clear_session()
    if REGRESSION_FLAG:
        R2_temp, super_scr = super_transfer(eur_model, EA, Minor, **para_mix)
        Map['tl2'] = R2_temp
    else:
        auc_temp, pr_auc_temp, super_scr = super_transfer(eur_model, EA, Minor, **para_mix)
        Map['tl2'] = auc_temp
        Map['tl2_PR'] = pr_auc_temp
    # convert super_scr to a list make it equal to the temp_df_pred.shape[1]
    super_scr = super_scr.reshape(-1).tolist()
    super_scr = super_scr + [None]*(temp_df_pred.shape[1] - len(super_scr)-1)+ ['tl2']
    temp_df_pred.loc[len(temp_df_pred)] = super_scr
    if REGRESSION_FLAG:
        del eur_model, R2_temp, super_scr
    else:
        del eur_model, auc_temp, pr_auc_temp, super_scr
    gc.collect()
    K.clear_session()
    df = pd.DataFrame(Map, index=[seed])
    temp_df_pred['seed'] = seed
    print(df)
    return df, temp_df_pred

# py neural_network_exisitng.py 0.5 0.9 0.2 'C:/Users/skumar26/CHR_1_AFR_EUR_synthetic/' 'EUR_chr1' 'AFR_chr1' 0
##########################################################################################################
def main():
    np.random.seed(1)
    # read command line arguments
    ddp_str= sys.argv[1]
    h_sq_list = [float(sys.argv[2])]
    print('h_sq_list', h_sq_list)
    rho_list = [float(sys.argv[3])]
    lambda__list = [0.1, 0.2, 0.3, 0.4, 0.5]
    file_path = './'
    data_eur_main = pd.read_csv('data_eur.csv').iloc[3:,:].astype(float)
    if os.path.exists('data_'+ddp_str+'_flipped.csv'):
        data_ddp_main = pd.read_csv('data_'+ddp_str+'_flipped.csv').iloc[3:,:].astype(float)
    else:
        if ddp_str=='eur':
            data_ddp_main = data_eur_main.copy()
    # create the directory
    new_dir = file_path+ddp_str
    os.makedirs(new_dir, exist_ok=True)
    all_training_fig = file_path+'/'+ddp_str+'/val_results_'
    os.makedirs(all_training_fig, exist_ok=True)
    for h_sq in h_sq_list:
        for rho in rho_list:
            for lambda_ in lambda__list:
                pred_df= pd.DataFrame()
                for seed in range(40):
                    title_main = 'h_sq_' + str(h_sq) + '_rho_' + str(rho) + '_lambda_' + str(lambda_) + ddp_str
                    # generate case and control data for euro and ddp data
                    #print data_eur_main data type
                    if ddp_str == 'amr':
                        race_seed = 101
                    elif ddp_str == 'sas':
                        race_seed = 202
                    elif ddp_str == 'eas':
                        race_seed = 303
                    elif ddp_str == 'afr':
                        race_seed = 404
                    elif ddp_str == 'eur':
                        race_seed = 202
                    data_eur, data_ddp = generate_case_control_data(h_sq, rho, data_eur_main.copy(), 
                                                                    data_ddp_main.copy(), random_seed=seed, 
                                                                    race_seed= race_seed)                           
                    # concatenate the data
                    data = concat_euro_ddp(file_path, data_eur, data_ddp, ddp_str=ddp_str, lambda_=lambda_)
                    # get the data for all models
                    data = get_data_for_all_models(data, ddp_str = ddp_str, seed=seed)
                    df1, temp_df_pred = run_cv(data,ddp_str, seed, train_fig_paths= all_training_fig+'/'+title_main+"_")
                    if seed == 0:
                        pred_df = temp_df_pred
                    else:
                        pred_df = pd.concat([pred_df, temp_df_pred], axis=0)
                    del temp_df_pred
                    gc.collect()
                    K.clear_session()
                    
                    if seed == 0:
                        df_res = df1
                    else:
                        df_res = pd.concat([df_res, df1])
                    del data_eur, data_ddp, data, df1
                    gc.collect()
                print('df_res \n', df_res)
                # save the prediction data as a parquet file
                # pred_df.to_parquet(new_dir+'/pred_' + title_main + '.parquet', index=False)
                # to csv
                pred_df.to_csv(new_dir+'/pred_' + title_main + '.csv', index=False)
                # save the results
                df_res.to_csv(new_dir+'/results_' + title_main + '.csv')
                # delete the data
                del df_res, pred_df
                gc.collect()
    del data_eur_main, data_ddp_main
    gc.collect()
    



if __name__ == '__main__':
    main()
# py neural_network_exisitng_Layernorm.py 'afr' 0.5 0.75
        
