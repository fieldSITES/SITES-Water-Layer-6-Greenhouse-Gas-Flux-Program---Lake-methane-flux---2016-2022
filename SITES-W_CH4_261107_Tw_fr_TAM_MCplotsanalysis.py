# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:54:08 2025

@author: David Bastviken

Using Monte-Carlo extrapolation output to generate boxplots, integrated 
yearly data and making statistical tests.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests
import itertools


### MAKE VIOLIN PLOTS ###

# Get MC output data
dfMC_F_lke_Y = pd.read_pickle("dfMC_F_lke_Y.pkl")

# LAKES def
ALM = "Almberga"
ERK = "Erken"
STO = "Stortjärn"
FER = "Feresjön"
ERS = "Erssjön"
lake = [ALM, ERK, STO, FER, ERS]  # Lake df

# Exclude 2023
MCvprep = dfMC_F_lke_Y.copy()
MCvprep = dfMC_F_lke_Y[dfMC_F_lke_Y['Y'] != 2023]
# NOTE Unit for flux is mol lake-1 yr-1
MCvprep['F_lke_kgCH4'] = MCvprep['F_lake'] * 16.04/1000
MCvprep['F_lke_tnCH4'] = MCvprep['F_lake'] * 16.04/1000000 

F_lke = 'F_lke_tnCH4'
  
for i in range(len(lake)):
    MCvplot_i = MCvprep.loc[MCvprep['lake'] == i]   
    dfplt = MCvplot_i[['Y', F_lke]]  
    # df_long = prepare_long_format(df, value_col= F_lke)
    dfplt['Yr'] = dfplt['Y'].astype(str)  # Convert to string for categorical x-axis
 # year as string to make categorical x-axis

    plt.figure(figsize=(6, 6))
    sns.boxplot(x='Yr', y= F_lke, data=dfplt, width=0.5, color='grey')
    
    plt.title(lake[i], fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Ftot (ton CH4 lake-1 yr-1', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Group data by year
    groups = [group[F_lke].dropna().values for _, group in dfplt.groupby('Yr')]
    
    # Perform the Kruskal-Wallis H-test
    h_stat, p_value = kruskal(*groups)

    print("H-statistic:", h_stat)
    print("p-value:", p_value)
    
    ## KW Post-Hoc

    # a) Create groups combinations in pairs (A vs B, A vs C, B vs C)
    dfplt['Yr_str'] = dfplt['Yr'].astype(str)
    groups_yr = dfplt['Yr_str'].unique()
    paired_combinations = list(itertools.combinations(groups_yr, 2))

    # b) Loop pairs and run Mann-Whitney U-test
    raw_p_values = []
    pair_names = []

    for group1, group2 in paired_combinations:
        data1 = dfplt[dfplt['Yr_str'] == group1][F_lke]
        data2 = dfplt[dfplt['Yr_str'] == group2][F_lke]
    
    # c) Run Mann-Whitney U-test
        _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
    
        raw_p_values.append(p_val)
        pair_names.append(f"{group1} vs {group2}")

    # d) Correct for multiple comparisons with statsmodels (Holm-Bonferroni)
    # 'holm' (Holm-Bonferroni) recommended before ordinary Bonferroni because of higher power
    rejectH0, adj_p_values, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')

    # e) Results
    KW_PostHoc_results = pd.DataFrame({
        'Comparison': pair_names,
        'raw_p_values': raw_p_values,
        'Adj_p-value': adj_p_values,
        'Sign_diff?': rejectH0
    })

    print(KW_PostHoc_results)




