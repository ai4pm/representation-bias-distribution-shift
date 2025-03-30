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
# suppress warnings
import warnings
warnings.filterwarnings("ignore")
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, "darkblue"), (0.5, "white"), (1, "darkred")]
len_palette = 110
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=len_palette)
custom_palette = [custom_cmap(i / (len_palette-1)) for i in range(len_palette)]

h_sq_list = [0.25, 0.5]
rho_list = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
lambda__list = [0.1, 0.2, 0.3, 0.4, 0.5]
dict_gap = {}
metric1='_sq_corr'
metric2=''
dict_gap[f'Mix1{metric1}']= [f'Mix1{metric2}']
dict_gap[f'ind_1{metric1}']= [f'ind_1{metric2}']
dict_gap[f'Mix2{metric1}']= [f'Mix2{metric2}']
dict_gap[f'ind_2{metric1}']= [f'ind_2{metric2}']
dict_gap[f'naive{metric1}'] = [f'naive{metric2}']
dict_gap[f'tl2{metric1}'] = [f'tl2{metric2}']
# dict_gap[f'tl2_Mixed{metric1}'] = [f'tl2_Mixed{metric2}']


def heatmap(x, y, fig, ax, ax_bar = None,  **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            # val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            # val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            # return val_position * size_scale
            return val * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    # plot_grid = plt.GridSpec(1, 25, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    # ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if ax_bar is not None:
        if color_min < color_max:
        # ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot
            ax = ax_bar

            col_x = [0]*len(palette) # Fixed x coordinate for the bars
            bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

            bar_height = bar_y[1] - bar_y[0]
            ax.barh(
                y=bar_y,
                width=[5]*len(palette), # Make bars 5 units wide
                left=col_x, # Make bars start at 0
                height=bar_height,
                color=palette,
                linewidth=0
            )
            ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
            ax.grid(False) # Hide grid
            ax.set_facecolor('white') # Make background white
            ax.set_xticks([]) # Remove horizontal ticks
            ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
            ax.yaxis.tick_right() # Show vertical ticks on the right 
    # save the plot
    plt.savefig('C:/Users/skumar26/synthetic_data_Generation_RESULTS/Gaps_plots/' + 'temp' + '.png')
    # if ax_bar is not None:
    #     return fig, ax, ax_bar
    # else:
    #     return fig, ax


# make a new directory to save the plots pf gaps
os.makedirs('C:/Users/skumar26/synthetic_data_Generation_RESULTS/Gaps_plots', exist_ok=True)
saving_title = f'{metric1.upper()}_Vs_R2_neural_3000SNPs_100000P'
second_file_temp_prefix = 'neural_regression_random_genomic_val_test_together_40_runs_3000SNPs_100000_patients'
for temp_file_prefix in [
    'neural_regression_random_genomic_val_test_together_40_runs_3000SNPs_100000_patients',                        
    ]:
    print('FIRST FILE: '+ temp_file_prefix)

    for h_sq in h_sq_list:
        fig_col = 5
        fig_row = len(dict_gap.keys())
        fig_heat, axs = plt.subplots(nrows=fig_row, ncols=fig_col, figsize=(20, 10),
                            gridspec_kw={
                                'width_ratios': [10, 10,10,10, 1],
                            'wspace': 0.3,
                            'hspace': 0.3})
        heat_col= 0
        for ddp_str in ['AMR', 'SAS', 'EAS', 'AFR']:
            df_3d = pd.DataFrame(columns=['ddp_str', 'rho', 'lambda', 'key', 'median_gap', 'std_gap', 'pvalue'])  
            row_idx = 0
            for key in dict_gap.keys():
                for rho in rho_list:
                    for lambda_ in lambda__list:
                        file_name = 'h_sq_' + str(h_sq) + '_rho_' + str(rho) + '_lambda_' + str(lambda_) + ddp_str
                        file_name1 =  'C:/Users/skumar26/synthetic_data_Generation_RESULTS/'+ temp_file_prefix+f'/{ddp_str}/COMPILED_' + file_name + '.csv'
                        file_name2 = 'C:/Users/skumar26/synthetic_data_Generation_RESULTS/'+ second_file_temp_prefix+f'/{ddp_str}/results_' + file_name + '.csv'
                        # read file

                        df = pd.read_csv(file_name1)
                        df2 = pd.read_csv(file_name2)
                        assert not df.equals(df2)

                        temp_series = df[key] - df2[dict_gap[key]].median(axis=1)
                        # assert  their length is equal to each other
                        assert len(df) == len(df2)
                        # assert these series are not same
                        assert not df[key].equals(df2[dict_gap[key]].median(axis=1))
                        if df[key].isnull().values.any():
                            # print the file name
                            print(file_name1)
                        assert not df[key].isnull().values.any()
                        assert not df2[dict_gap[key]].median(axis=1).isnull().values.any()
                        assert not temp_series.isnull().values.any()

                        assert key == dict_gap[key][0]
                        if len(temp_series) != 40:
                            print('Error: length of temp_series is not 40')
                            exit()

                        pvalue = -1*math.log10(stats.mannwhitneyu(df[key], df2[dict_gap[key]].median(axis=1), alternative='greater')[1])
                        pvalue_opposite = -1*math.log10(stats.mannwhitneyu(df2[dict_gap[key]].median(axis=1), df[key], alternative='greater')[1])

                        if pvalue<1.302 and pvalue_opposite>1.302:
                            temp_series = -1*temp_series
                            pvalue = -pvalue_opposite
                        elif pvalue<1.3 and pvalue_opposite<1.3:
                            temp_series = 0*temp_series
                            pvalue = 0

                        df_3d.loc[row_idx] = [ddp_str, rho, lambda_, key[:-1].upper(), temp_series.median(), temp_series.std(), pvalue]
                        # find max and min size of the gap
                        del df
                        
                        # delete the data
                        del temp_series, pvalue
                        gc.collect()
                        row_idx += 1
            # plot heat map for each key, with values as gap and color as pvalue, where all hsq subplots in one figure
            for i, key in enumerate(dict_gap.keys()):
                df_temp = df_3d[df_3d['key']==key[:-1].upper()]

                if fig_row-1 and heat_col == fig_col-2:
                    heatmap(
                            df_temp['lambda'], df_temp['rho'], fig=fig_heat, 
                            ax=axs[i, heat_col], ax_bar=axs[i, heat_col+1],
                            color=df_temp['pvalue'], color_range=[-5, 5],
                            palette=sns.color_palette(custom_palette),
                            size=df_temp['median_gap'], size_range=[0,0.1],
                            marker='h',
                            size_scale=4000 ) 
                else:
                    heatmap(
                            df_temp['lambda'], df_temp['rho'], fig=fig_heat, 
                            ax=axs[i, heat_col], ax_bar=None,
                            color=df_temp['pvalue'], color_range=[-5, 5],
                            palette=sns.color_palette(custom_palette),
                            size=df_temp['median_gap'], size_range=[0,0.1],
                            marker='h',
                            size_scale=4000 )
                if i == fig_row-1:
                    axs[i, heat_col].set_xlabel('lambda')
                else:
                    axs[i, heat_col].set_xlabel('')
                # set title for each subplot on the first column
                if i == 0:
                    axs[i, heat_col].set_title('EUR/'+str(ddp_str))
                else:
                    axs[i, heat_col].set_title('')
                del df_temp,
                gc.collect()    
                # when heat_col is 0, then add extra axis to the left of the original one
                if heat_col == 0:
                    # replace Mix_1 with Mixture_Gap, ind_1 with Independent_Gap, tl2 with Transfer_Learning_Gap
                    if key == f'Mix1{metric1}':
                        key = 'Mix1_Vs_Mix1'
                    elif key == f'ind_1{metric1}':
                        key = 'Ind1_Vs_Ind1'
                    elif key == f'naive{metric1}':
                        key = 'Naive_Vs_Naive'
                    elif key == f'tl2{metric1}':
                        key = 'TL_Vs_TL'
                    elif key == f'Mix2{metric1}':
                        key = 'Mix2_Vs_Mix2'
                    elif key == f'ind_2{metric1}':
                        key = 'Ind2_Vs_Ind2'


                    # add extra axis to the left of the original one
                    ax2 = axs[i, heat_col].twinx()
                    # move extra axis to the left, with offset
                    ax2.yaxis.set_label_position('left')
                    ax2.spines['left'].set_position(('axes', -0.2))
                    # hide spine and ticks, set group label
                    ax2.spines['left'].set_visible(False)
                    ax2.set_yticks([])
                    ax2.set_ylabel(key, rotation=90, size='large',
                                ha='center', va='center', labelpad=20)
            del df_3d
            gc.collect()
            heat_col += 1   
        # overall title for the figure
        if metric1 == '':
            if 'regression' not in temp_file_prefix:
                fig_heat.suptitle('h_sq_'+str(h_sq)+'_AUROC', fontsize=16)
            else:
                fig_heat.suptitle('h_sq_'+str(h_sq)+'R2', fontsize=16)
        else:
            fig_heat.suptitle('h_sq_'+str(h_sq)+saving_title, fontsize=16)
        # save the figure
        fig_heat.savefig('C:/Users/skumar26/synthetic_data_Generation_RESULTS/Gaps_plots/Appealing_Heatmap_'+str(h_sq)+ temp_file_prefix+saving_title+'.png')
        plt.close('all')
        del fig_heat, axs, heat_col
        gc.collect()





