# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:44:51 2023

@author: David Bastviken

# Memory note for 3D rotability: %matplotlib qt

# 2023-11-25: Medel per djupzon för tempmodell. Namnge sjöar i plottar.
# 2025-05-13: Final data used
# 2026-09-24: Revised extrapolation and uncertainty analysis.

### --- SITES Water CH4 emissions 2016-2022 ---###
Script for main calculations of SITES lake CH4 flux and concentration patters. 
The script is numbered by sections to explain what different parts do. 
The operations performed includes:
    - Gathering, selecting and organizing data to use for different operations.
    - Defining lakes and lake characteristics such as depth zones and their areas.
    - Integration of fluxes across depth zones.
    - Integration of fluxes across the whole lake area accounting for the area of the different depth zones.
    - Establishing flux-temperature relationships based on best fit with empirical observations, 
    thereby also determining temperature-normalized fluxes and the temperature response factor.
    - Performing a Monte-Carlo extrapolation of fluxes across the years 2016-2022 
    including calculating among-year anomalies for each lake.
    - Generating an output pkl-file for the Monte-Carlo extrapolation for plotting and 
    statistical tests in separate script.
    - Making various plots.
    - IMPORTANT: Making critical selections - i.e. whether the run should focus on 
    total flux (Ftot), diffusive flux (Fdiff), or surface water CH4 concentrations (Caq), 
    which water temperatures to use, threshold temperatures in each lake to define 
    the open water season duration, and number of iterations in the Monte-Carlo extrapolation among years.  

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as stm
from matplotlib.ticker import StrMethodFormatter
import scipy.stats as stats

#########################################################

# (1) IMPORT SURFACE WATER TEMP (Tw)
# Mesan air temperatures = SMHI model to interpolate to grid coordinate between weather stations; see https://www.smhi.se/data/om-smhis-data/uppdateringar-oppna-data/uppdateringar-i-smhis-oppna-data/2024-01-18-ny-version-av-api-for-meteorologiska-analyser-mesan)
# Surface water temperature extimted from mean data in SITES-W_260923-mesanT-wT.py
Tw_fr_TAM = pd.read_pickle("Tw_fr_TAMrev.pkl")
Tw_fr_TAM['Time'] = pd.to_datetime(Tw_fr_TAM['Time']).astype('datetime64[ns]') #set time format

# (2) LAKES
ALM = "Almberga"
ERK = "Erken"
STO = "Stortjärn"
FER = "Feresjön"
ERS = "Erssjön"
lake = [ALM, ERK, STO, FER, ERS]  # Lake df

# (3) AREA proportions
# Tot area
Alake = [57884, 23682173, 39222, 516193, 67036]  # m2
# Area fraction
Af1 = [0.286365836, 0.312279732, 0.401354433]  # ANS
Af2 = [0.09805924099999999, 0.102654413, 0.799286346]  # ERK
Af3 = [0.43452114, 0.313432832, 0.252046027]  # STO approx from vol vs depth
Af4 = [0.26106127, 0.738939, 0]  # FER
Af5 = [0.423881496, 0.576118503, 0]  # ERS
Aflake = [Af1, Af2, Af3, Af4, Af5]

# (4)  Get FLUX data
# Find flux data file
path = r'D:\Arb_v4\Forskning\Projekt\SITES\SITES_CH4_2016-22_ms'
# File path on computer
fname = 'FCH4extract250513.xlsx'  # File name
# Define dataframe named "odata" from datafile and define columns and rows(index)
# Original data from excel file.
odata = pd.read_excel(r'%s\%s' % (path, fname))
col = odata.columns
ind = odata.index
# Version of original data (odata) used for calcs. Make copy to not modify original source df.
dfAll = odata.copy()
# Add mean water CH4 conc. (Mean between initial and end conc.)
dfAll['mnCH4aq_uM'] = (dfAll['CH4aq_in_uM'] + dfAll['CH4aq_end_uM'])/2
dfAll.set_index("Lake", inplace=True)
# Add column classifying depth zone.
dfAll["dz"] = pd.cut(x=dfAll["z_init_m"], bins=[0, 2, 4, np.inf],
                     labels=['0-2', '2-4', '>4'])
dfAll['dt_FC_h'] = dfAll['dt_FC_d']*24

## (5) Selecting variables to work with #################################
dfSome1 = dfAll.copy()[['Time_init', 'Time_end', 'z_init_m', 'PhPa_in', 'PhPa_end', 'dt_FC_d',
                        'dCH4_FC_umol', 'mnTwC', 'k_md-1', 'FCH4d_mmolm-2d-1', 'FCH4e_mmolm-2d-1',
                        'FCH4tot_mmolm-2d-1', 'dz', 'mnCH4aq_uM', 'dt_FC_h']] # Overkill - not all of these variables used.

# Preserve a column with original time after making 'Time_end' DateTimeIndex later.
dfSome1['Time'] = dfSome1['Time_end'].copy()
# Ensure DateTime format.
dfSome1['Time_init'] = pd.to_datetime(dfSome1['Time_init'])
dfSome1['Time'] = pd.to_datetime(dfSome1['Time'])  # Ensure DateTime format.
# Create approximate DOY column for later use.
dfSome1['DOY'] = dfSome1['Time_init'].dt.dayofyear
dfSome1['Time'] = pd.to_datetime(dfSome1['Time']).astype('datetime64[ns]') #set time format

# (6) Definitions; Selecting variables in focus

Caq = 'mnCH4aq_uM'
k_app = 'k_md-1'
Fdiff = 'FCH4d_mmolm-2d-1'
Febul = 'FCH4e_mmolm-2d-1'
# Fdiff = diffusive flux, Febul = ebullition, Ftot = total flux
Ftot = 'FCH4tot_mmolm-2d-1'
TwC = 'mnTwC'
# TwC = water temp measured at chamber,  Tw = water temp from mesan air temperatures (TAM)
Tw = 'Tw_fr_TAM'

# **************************************
# ****************************************
# Select VARIABLE in focus here (Ftot, Fdiff, Febul or Caq). "F" denote "variable in Focus"
F = Ftot
T = Tw  # Select TEMPERATURE TYPE in focus here.
# ***************************************
# ***************************************

## (7) Basic PLOTS and SETUP for depth zone normalization ##############################

# Depth zones; Few measurements in 0-1 m = merge to 1-2 m.
dzone = ['0-2', '2-4', '>4']

dfAfi = []
dfLdzAll = []  # by lake, depth zone and date; all values
dfLdzmnAll = []  # by lake, depth zone and date; mean values, all data
dfLdzmnSa = []  # by lake, depth zone and date; mean values, subset
corrTall = []
dfLdzmnMa = []
RawDtaAll = []
RawDtaAlldz = []

for i in range(len(lake)):
    dflake = dfSome1.loc[[lake[i]]]
    Numdz = dflake['dz'].value_counts()
    Ndz = Numdz[Numdz != 0]  # drop zero rows (depth zones with no data)

# Plot versus DOY
    DOYplt = dflake.copy()[['DOY', F]]
    DOYplt = DOYplt.dropna()
    DOYplt.plot.scatter(x="DOY", y=F, )
    plt.yticks(fontsize=15)
    plt.ylabel(F, fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel('DOY', fontsize=15)
    plt.suptitle(lake[i])
    plt.show

# Plot vs time AND raw mean F or conc
    timeplt = dflake.copy()[['Time_init', F]]
    timeplt = timeplt.dropna()

# Generating Table data overview - descriptives
    NumF = timeplt[F].count()
    MinF = timeplt[F].min()
    MaxF = timeplt[F].max()
    AvgF = timeplt[F].mean()
    SD_F = timeplt[F].std()
    # to be converted to dataframe
    RawDta = [lake[i], NumF, MinF, MaxF, AvgF, SD_F]
    dfRawDta = pd.DataFrame(RawDta).T
    dfRawDta.columns = ["lake", "NumF", "MinF", "MaxF", "AvgF", "SD_F"]
    RawDtaAll.append(dfRawDta)

    timeplt.plot.scatter(x='Time_init', y=F)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Tick marks at Jan 1
    plt.xlabel('Time', fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel(F, fontsize=15)
    #plt.ylabel(r"$\mathrm{CH}_{\scriptstyle 4} \ " r"(\mathrm{mmol} \ \mathrm{m}^{\scriptstyle -2} \ \mathrm{d}^{\scriptstyle -1})$", fontsize=15) 
    plt.xticks(fontsize=15)
    plt.suptitle(lake[i])
    # plt.tight_layout()
    plt.show

## (8) CALCULATE INTEGRATED FLUX FROM COMBINED DEPTH ZONES ####################################################

    dfAfi = Aflake[i]  # Depth zone areas for Lake i

    for j in range(len(Ndz)):
        dflake_dzc = dflake[(dflake['dz'] == dzone[j])]
        # Make copy to not modify original source df.
        dflake_dzp = dflake_dzc.copy()
        # Set DateTime to index to allow DateTime slicing
        dflake_dzp = dflake_dzp.set_index('Time')
        dfLdzAll.append(dflake_dzp)  # All data by lake and depth zone

# Resample means for all variables by hour and depth zone - for aligning by time with mesan temp data, and then remove Nan and resample by day.
        dflake_dzhmn = dflake_dzp.resample('h')['z_init_m', 'PhPa_in', 'PhPa_end', 'dt_FC_d', 'dCH4_FC_umol',
                                                'mnTwC', 'k_md-1', 'FCH4d_mmolm-2d-1', 'FCH4e_mmolm-2d-1', 'FCH4tot_mmolm-2d-1',
                                                'mnCH4aq_uM', 'DOY', 'dt_FC_h'].mean()
        dflake_dzhmn.index.rename('TIMESTAMP', inplace=True)
# Prepare temp from mesan data for merger.
        Tw_fr_TAMi = Tw_fr_TAM.loc[Tw_fr_TAM['lake_no'] == i]
        Tw_fr_TAMi = Tw_fr_TAMi.set_index('Time')
        Tw_fr_TAMi.index.rename('TIMESTAMP', inplace=True)
# Integrate Tw by deployment time period
        dt_deploy = dflake_dzhmn[['dt_FC_h']].copy()
# Convert period Pd (decimal hours) into Timedelta format
        dt_deploy['Pd_dt_h'] = pd.to_timedelta(dt_deploy['dt_FC_h'], unit='h')  # Converts hours to timedelta
# Merge with Tw_fr_TAMi
        Tw_fr_TAMiTw = pd.merge_asof(dt_deploy, Tw_fr_TAMi, on='TIMESTAMP')
        Tw_fr_TAMiTw = Tw_fr_TAMiTw.set_index('TIMESTAMP')
# Function to calculate rolling mean over historical period Pd
        def compute_historical_mean(row, Tw_fr_TAMiTw):
            if pd.notna(row['Pd_dt_h']):
                time_window = pd.Timestamp(row.name) - row['Pd_dt_h']  # Defines start of the lookback period # Convert row index to Timestamp
                return Tw_fr_TAMiTw.loc[time_window:row.name, 'Tw'].mean()  # Computes mean over time window
            return pd.NA  # Leave NaT rows untouched
# Apply function to rows where Pd exists
        Tw_fr_TAMiTw['roll_Tw'] = Tw_fr_TAMiTw.apply(lambda row: compute_historical_mean(row, Tw_fr_TAMiTw), axis=1)
# Merge by hour
        dflake_dzMh = pd.merge_asof(dflake_dzhmn, Tw_fr_TAMiTw, on='TIMESTAMP')  # Match hours
        dflake_dzMh = dflake_dzMh.set_index('TIMESTAMP') # Renew timestamp index.
        # The merger leads to two DOY colum,ns - DOY_x and DOy_y. Kept for check but new DOY col created.
        dflake_dzMh['DOY'] = dflake_dzMh['DOY_x']
# Resample by day
        dflake_dzMD = dflake_dzMh.resample('D')['z_init_m', 'PhPa_in', 'PhPa_end', 'Pd_dt_h', 'dt_FC_d','dt_FC_h_x', 'dCH4_FC_umol',
                                                'mnTwC', 'k_md-1', 'FCH4d_mmolm-2d-1', 'FCH4e_mmolm-2d-1', 'FCH4tot_mmolm-2d-1',
                                                'mnCH4aq_uM', 'Tw', 'roll_Tw', 'DOY'].mean()
        dflake_dzmn = dflake_dzMD.copy().dropna()
        dflake_dzmn['roll_Tw'] = dflake_dzmn['roll_Tw'].astype(float) # convert objet to float ensuring no extra spaces around values

        Afij = dfAfi[j]  # AREA FRACTION for depth zones and lakes
# Flux * Area fraction per each depth zone
        dflake_dzmn['F_Afij'] = dflake_dzmn[F]*Afij
# Conc * Area fraction per each depth zone
        dflake_dzmn['Caq_Afij'] = dflake_dzmn[Caq]*Afij
        dflake_dzmn['dz'] = dzone[j]
        dflake_dzmn['lake'] = lake[i]
        dfLdzmnAll.append(dflake_dzmn) # mean for all lakes by depth zone and day
# Selected subset for later calcs (when integrating depth zone data)       
        dfLdzmnS = dflake_dzmn[['roll_Tw','F_Afij','Caq_Afij', 'DOY']]
        dfLdzmnSa.append(dfLdzmnS)
# Grand mean per lake and depth zone all times
        NumFdz = dflake_dzmn[F].count()
        NumCdz = dflake_dzmn[Caq].count()
        MinFdz = dflake_dzmn[F].min()
        MinCdz = dflake_dzmn[Caq].min()
        MaxFdz = dflake_dzmn[F].max()
        MaxCdz = dflake_dzmn[Caq].max() 
        AvgFdz = dflake_dzmn[F].mean()
        AvgCdz = dflake_dzmn[Caq].mean()
        SD_Fdz = dflake_dzmn[F].std()
        SD_Cdz = dflake_dzmn[Caq].std()
        RawDtadz = [lake[i], dzone[j], NumCdz, MinCdz, MaxCdz, AvgCdz, SD_Cdz, NumFdz, MinFdz, MaxFdz, AvgFdz, SD_Fdz] # to be converted to dataframe
        dfRawDtadz = pd.DataFrame(RawDtadz).T
        dfRawDtadz.columns = ["lake", "dz", "NumCdz", "MinCdz", "MaxCdz", "AvgCdz", "SD_Cdz", "NumFdz", "MinFdz", "MaxFdz", "AvgFdz", "SD_Fdz"]
        RawDtaAlldz.append(dfRawDtadz)
# Gather data for all lakes and times
dfLdzmnA = pd.concat(dfLdzmnAll) # All variables  
dfStatAll = pd.concat(RawDtaAll)
dfStatAlldz = pd.concat(RawDtaAlldz)


## (9) TEMPERATURE RELATIONSHIPS BY DEPTH ZONE FROM MESAN TEMPERATURES ###########################################

# Select PLOT DATA for below 
dfPlot = dfLdzmnA
# Prepare dataframes
TrangeA = []
dfvardz = []
BrPardz = []
TwR = 'roll_Tw'
# By lake...
for g in range(len(lake)):     
    dfg = dfPlot.loc[dfPlot['lake'] == lake[g]].copy()
    Numdzg = dfg['dz'].value_counts()
    Ndzg = Numdzg[Numdzg != 0] #drop zero rows
# Dataframes by depth zone    
    df0 = dfg.loc[dfg['dz'] == '0-2']
    df1 = dfg.loc[dfg['dz'] == '2-4']
    df2 = dfg.loc[dfg['dz'] == '>4']
    
# Identify temperature ranges
    MinTwR = dfg[TwR].min()
    MaxTwR = dfg[TwR].max()
    Trange = [lake[g], MinTwR, MaxTwR] # to be converted to dataframe
    dfTrange = pd.DataFrame(Trange).T
    dfTrange.columns = ["lake", "MinTwR", "MaxTwR"]
    TrangeA.append(dfTrange)
    
## CURVE FIT F VS TEMP BY DEPTH ZONE

# Curve fit equation F vs temp by depth zone
    for h in range(len(Ndzg)):
        dfh = dfg[(dfg['dz'] == dzone[h])]
# Equation        
        def func(xdata, a, b):
            return (a * b**(xdata-20)) # Temp eq from Aben et al. 2017.
# Data and initial guess
        xdata = dfh[TwR]
        # guess = [1, 1]  # Guessed starting values of a,b
        param_bounds = ([0.01, 0.1],[10, 1.35]) # parameter boundaries. Erken 0-2 m goes crazy without boundaries.
        ydata = dfh[F]
        popt, pcov = curve_fit(func, xdata, ydata, bounds = param_bounds)
# Curve fit evaluation vs the data
        modF = func(xdata, *popt)
        Resid = ydata - modF  # Residuals versus Xvar
        RMSE = [np.sqrt(np.mean(Resid ** 2))] # RMSE
        dfRMSE = pd.DataFrame(RMSE)
        SD = np.sqrt(np.diag(pcov))
        poptdf = pd.DataFrame(popt).T  # convert popt to dataframe; .T is for transposing
        vardf = pd.DataFrame(SD).T
        poptvar = pd.concat([poptdf, vardf], axis=1)
        poptvar['lke'] = lake[g]
        poptvar['dzn'] = dzone[h]
        dfvardz.append(poptvar)
# Pred vs Obs
        Xpvo = pd.DataFrame(modF)
        Xpvo.columns = ["modF"]
        Xpvo = stm.add_constant(Xpvo)
        model = stm.OLS(ydata, Xpvo)
        Brts = model.fit()
        Brts_h = pd.DataFrame(Brts.params).T
        Brsq_h = pd.DataFrame([Brts.rsquared_adj])
        BrSE_h = pd.DataFrame(Brts.bse).T
        BrPred = pd.DataFrame(Brts.predict())
        BrPval = pd.DataFrame(Brts.pvalues)
        BrPva = BrPval.T
        BrPconf = pd.DataFrame(Brts.conf_int())
        BrConf = pd.DataFrame([BrPconf.iloc[0,0], BrPconf.iloc[1,0], BrPconf.iloc[0,1], BrPconf.iloc[1,1]]).T

        BrPar_h = pd.concat([Brts_h, Brsq_h, BrSE_h, BrPva, BrConf, dfRMSE], axis=1)
        BrPar_h['lke'] = lake[g]
        BrPar_h['dzn'] = dzone[h]
        BrPardz.append(BrPar_h)        
# Pred vs Obs - intercept, slope, SE, p confidence interval RMSE...
    Br_dzg = pd.concat(BrPardz, ignore_index=True)
    Br_dzg.columns = (
        ['c', 'f', 'R2', 'SEc', 'SEf', 'pc', 'pf', 'conflow_c', 'confhigh_c', 'conflow_f', 'confhigh_f', 'RMSE','lke','dzn']) # Pred vs Obs stats
        # c is constant; f is factor
# Temperature response coefficients    
    Bdf_coef_dzg = pd.concat(dfvardz, ignore_index=True)
    Bdf_coef_dzg.columns = (['a', 'b', 'asd', 'bsd','lke','dzn']) # Coefficients with sd
    
    # BrPred_dzg = pd.concat(BrPred, ignore_index=True)
    
## (10) PLOTS by dz and lake & points plus equation line #####
    coef0 = Bdf_coef_dzg.loc[(Bdf_coef_dzg['lke'] == lake[g]) & (Bdf_coef_dzg['dzn'] == '0-2')]
    coef1 = Bdf_coef_dzg.loc[(Bdf_coef_dzg['lke'] == lake[g]) & (Bdf_coef_dzg['dzn'] == '2-4')]
    coef2 = Bdf_coef_dzg.loc[(Bdf_coef_dzg['lke'] == lake[g]) & (Bdf_coef_dzg['dzn'] == '>4')]
    
    Lx0 = np.linspace(df0[TwR].min(), df0[TwR].max(), 100)
    Ly0 = coef0['a'].values[0] * coef0['b'].values[0]**(Lx0-20)

    Lx1 = np.linspace(df1[TwR].min(), df1[TwR].max(), 100)
    Ly1 = coef1['a'].values[0] * coef1['b'].values[0]**(Lx1-20)

    if len(coef2) > 0:
        Lx2 = np.linspace(df2[TwR].min(), df2[TwR].max(), 100)
        Ly2 = coef2['a'].values[0] * coef2['b'].values[0]**(Lx2-20)
    else:
        Lx2 = []
        Ly2 = []
        
    plt.figure()
    plt.subplot(111)
    plt.plot(df0[TwR], df0[F], 'o', ms=9, color = 'cornflowerblue')
    plt.plot(Lx0, Ly0, '-', color = 'cornflowerblue')
    plt.plot(df1[TwR], df1[F], 'x', ms=9, color = 'chocolate')
    plt.plot(Lx1, Ly1, '--', color ='chocolate')
    plt.plot(df2[TwR], df2[F], 'ks', markerfacecolor='none', ms=9, markeredgecolor='black')
    plt.plot(Lx2, Ly2, ':', color = 'black')
    plt.yticks(fontsize=20)
    plt.ylabel(F, fontsize=11)
    plt.xticks(fontsize=20)
    plt.xlabel('T (C)', fontsize=11)
    plt.suptitle(lake[g])
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
    plt.show()

TrangeAll = pd.concat(TrangeA) # Temperture ranges for different temperature measures.

## (11) Lake-integrated flux/m2 AND conc CH4      ############################################################  

ALM_mg1 = pd.merge(dfLdzmnSa[0],dfLdzmnSa[1], how='inner', on='TIMESTAMP')
ALM_mg2 = pd.merge(ALM_mg1 ,dfLdzmnSa[2], how='inner', on='TIMESTAMP')
ALM_mg2.columns = (['0TwR','0Faf','0CH4aq','0DOY', '1TwR','1Faf','1CH4aq','1DOY', '2TwR','2Faf','2CH4aq','2DOY'])
ALMF = ALM_mg2[['0Faf', '1Faf', '2Faf']].sum(axis=1) # sum of F * Area fraction for each depth zone = Area normalized flux (mmol m-2 d-1)
ALMTwR = ALM_mg2[['0TwR','1TwR', '2TwR']].mean(axis=1)
ALMCaq = ALM_mg2[['0CH4aq','1CH4aq', '2CH4aq']].sum(axis=1)
ALMdoy = ALM_mg2[['0DOY','1DOY', '2DOY']].mean(axis=1)
ALMc = pd.concat([ALMTwR, ALMF,ALMCaq, ALMdoy], axis=1)
ALMc.columns = (['TwR', 'FLn', 'CH4aq','DOY'])
ALMc['lake'] = 0

ERK_mg1 = pd.merge(dfLdzmnSa[3],dfLdzmnSa[4], how='inner', on='TIMESTAMP')
ERK_mg2 = pd.merge(ERK_mg1 ,dfLdzmnSa[5], how='inner', on='TIMESTAMP')
ERK_mg2.columns = (['0TwR','0Faf','0CH4aq','0DOY', '1TwR','1Faf','1CH4aq','1DOY', '2TwR','2Faf','2CH4aq','2DOY'])
ERKF = ERK_mg2[['0Faf', '1Faf', '2Faf']].sum(axis=1) # sum of F * Area fraction for each depth zone = Area normalized flux (mmol m-2 d-1)
ERKTwR = ERK_mg2[['0TwR','1TwR', '2TwR']].mean(axis=1)
ERKCaq = ERK_mg2[['0CH4aq','1CH4aq', '2CH4aq']].sum(axis=1) # sum of C * Area fraction for each depth zone = Area normalized conc (mmol m-3 = uM)
ERKdoy = ERK_mg2[['0DOY','1DOY', '2DOY']].mean(axis=1)
ERKc = pd.concat([ERKTwR, ERKF,ERKCaq, ERKdoy], axis=1)
ERKc.columns = (['TwR', 'FLn', 'CH4aq','DOY'])
ERKc['lake'] = 1

STO_mg1 = pd.merge(dfLdzmnSa[6],dfLdzmnSa[7], how='inner', on='TIMESTAMP')
STO_mg2 = pd.merge(STO_mg1 ,dfLdzmnSa[8], how='inner', on='TIMESTAMP')
STO_mg2.columns = (['0TwR','0Faf','0CH4aq','0DOY', '1TwR','1Faf','1CH4aq','1DOY', '2TwR','2Faf','2CH4aq','2DOY'])
STOF = STO_mg2[['0Faf', '1Faf', '2Faf']].sum(axis=1) # sum of F * Area fraction for each depth zone = Area normalized flux (mmol m-2 d-1)
STOTwR = STO_mg2[['0TwR','1TwR', '2TwR']].mean(axis=1)
STOCaq = STO_mg2[['0CH4aq','1CH4aq', '2CH4aq']].sum(axis=1)
STOdoy = STO_mg2[['0DOY','1DOY', '2DOY']].mean(axis=1)
STOc = pd.concat([STOTwR, STOF,STOCaq, STOdoy], axis=1)
STOc.columns = (['TwR', 'FLn', 'CH4aq','DOY'])
STOc['lake'] = 2

FER_mg1 = pd.merge(dfLdzmnSa[9],dfLdzmnSa[10], how='inner', on='TIMESTAMP')
FER_mg1.columns = (['0TwR','0Faf','0CH4aq','0DOY', '1TwR','1Faf','1CH4aq','1DOY'])
FERF = FER_mg1[['0Faf', '1Faf']].sum(axis=1) # sum of F * Area fraction for each depth zone = Area normalized flux (mmol m-2 d-1)
FERTwR = FER_mg1[['0TwR','1TwR']].mean(axis=1)
FERCaq = FER_mg1[['0CH4aq','1CH4aq']].sum(axis=1)
FERdoy = FER_mg1[['0DOY','1DOY']].mean(axis=1)
FERc = pd.concat([FERTwR, FERF,FERCaq, FERdoy], axis=1)
FERc.columns = (['TwR', 'FLn', 'CH4aq','DOY'])
FERc['lake'] = 3

ERS_mg1 = pd.merge(dfLdzmnSa[11],dfLdzmnSa[12], how='inner', on='TIMESTAMP')
ERS_mg1.columns = (['0TwR','0Faf','0CH4aq','0DOY', '1TwR','1Faf','1CH4aq','1DOY'])
ERSF = ERS_mg1[['0Faf', '1Faf']].sum(axis=1) # sum of F * Area fraction for each depth zone = Area normalized flux (mmol m-2 d-1)
ERSTwR = ERS_mg1[['0TwR','1TwR']].mean(axis=1)
ERSCaq = ERS_mg1[['0CH4aq','1CH4aq']].sum(axis=1)
ERSdoy = ERS_mg1[['0DOY','1DOY']].mean(axis=1)
ERSc = pd.concat([ERSTwR, ERSF,ERSCaq, ERSdoy], axis=1)
ERSc.columns = (['TwR', 'FLn', 'CH4aq','DOY'])
ERSc['lake'] = 4

FLnT = pd.concat([ALMc, ERKc, STOc, FERc, ERSc])

## (12) TEMP RELATIONSHIP FOR INTEGRATED WHOLE-LAKE FLUX ########################################
LGstats = []
FLmn = []
dfvar = []
BrPar = []
F_lake_Y = []
# lake = [ALM, ERK, STO, FER, ERS]  # Memory note from above on lake df.

#********* Defining open water season duration *******
lowTw_by_lake = [6, 3, 5, 5, 2] # low threshold Tw yielding mean DOY open-water matching 3-8 days shorter than observations.
#*********

for k in range(len(lake)):
    dfFLn = FLnT.loc[FLnT['lake'] == k]
    FLkTwRmin = dfFLn['TwR'].min()    
    FLkmin = dfFLn['FLn'].min() #lake grand min
    FLkmax = dfFLn['FLn'].max()
    FLkmn = dfFLn['FLn'].mean()
    FLksd = dfFLn['FLn'].std()
    CLkmin = dfFLn['CH4aq'].min()
    CLkmax = dfFLn['CH4aq'].max()
    CLkmn = dfFLn['CH4aq'].mean()
    CLksd = dfFLn['CH4aq'].std()
    LGmnk = ([lake[k], FLkTwRmin, FLkmin, FLkmax, FLkmn, FLksd, CLkmin, CLkmax, CLkmn, CLksd])
    LGstats.append(LGmnk)
    dfLGstats = pd.DataFrame(LGstats) # Combine data that are whole lake and depth-zone normalized
    dfLGstats.columns = ["lake", "Tmin", "minFLn","maxFLn", "AvgFLn", "SDFLn", "MinCdz", "MaxCdz", "AvgCdz", "SD_Cdz"]
# Curve fit equation F vs temp...
    def func(xdata, a, b):
        return (a * b**(xdata-20)) # Temp eq from Aben et al. 2017.
    xdata = dfFLn['TwR']
    guess = [1, 1]  # Guessed starting values of a,b
    ydata = dfFLn[['FLn']+['DOY']]
    popt, pcov = curve_fit(func, xdata, ydata['FLn'], guess)

# Modeled CH4
    modF = func(xdata, *popt)

    Resid = ydata['FLn'] - modF  # Residuals versus Xvar
    RMSE = [np.sqrt(np.mean(Resid ** 2))]
    dfRMSE = pd.DataFrame(RMSE)
    SD = np.sqrt(np.diag(pcov))
    poptdf = pd.DataFrame(popt).T  # convert ppopt to dataframe; .T is for transposing
    vardf = pd.DataFrame(SD).T
    poptvar_k = pd.concat([poptdf, vardf], axis=1)
    poptvar_k['lke'] = lake[k]
    poptvar_k.columns = (['a', 'b', 'asd', 'bsd','lake'])
    dfvar.append(poptvar_k)

# Pred vs Obs
    Xpvo = pd.DataFrame(modF)
    Xpvo.columns = ["modF"]
    Xpvo = stm.add_constant(Xpvo)
    model = stm.OLS(ydata['FLn'], Xpvo)
    Brts = model.fit()
    Brts_k = pd.DataFrame(Brts.params).T
    Brsq_k = pd.DataFrame([Brts.rsquared_adj])
    BrSE_k = pd.DataFrame(Brts.bse).T
    BrPred = pd.DataFrame(Brts.predict())
    BrPval = pd.DataFrame(Brts.pvalues)
    BrPva = BrPval.T
    BrPconf = pd.DataFrame(Brts.conf_int())
    BrConf = pd.DataFrame([BrPconf.iloc[0,0], BrPconf.iloc[1,0], BrPconf.iloc[0,1], BrPconf.iloc[1,1]]).T

    BrPar_k = pd.concat([Brts_k, Brsq_k, BrSE_k, BrPva, BrConf, dfRMSE], axis=1)
    BrPar_k['lke'] = lake[k]
    BrPar.append(BrPar_k)

    # print(Brts.summary())

## (13) PREPARATIONS FOR MONTE CARLO ##################################    
# Water temperatures for all days by lake
    dfTw_k = Tw_fr_TAM.loc[Tw_fr_TAM['lake_no'] == k].copy()
    dfTw_k['Time'] = pd.to_datetime(dfTw_k['Time']) # Ensure datatime format
    dfTw_k = dfTw_k.set_index('Time') # Make datatime to index
    dfTw_k_mnD = dfTw_k.resample('D')['DOY', 'lake_no', 'Tw', 'TAM'].mean() # Resample by day.    
    dfTw_k_mnD['Tw'] = dfTw_k_mnD['Tw'].astype(float) # convert objet to float ensuring no extra spaces around values 
# Select days above low temp threshold  
    MinTw_k = TrangeAll.loc[TrangeAll['lake'] == lake[k]].copy() # Select lake "k"
    minTw = MinTw_k._get_value(0, 'MinTwR')
# dfTw_k_rng = dfTw_k_mnD.loc[dfTw_k_mnD['Tw'] >= minTw] # Identify rows when temp >= minimum temp of observations
    dfTw_k_rng = dfTw_k_mnD.loc[dfTw_k_mnD['Tw'] >= lowTw_by_lake[k]] # Identify rows when temp >= minimum selected temp
    dfTw_k_use = dfTw_k_rng.copy()
# Flux calc parameter distributions
    mean_a = poptvar_k.loc[0,'a'] # Example mean of parameter
    SD_a = poptvar_k.loc[0,'asd']  # Example standard deviation
    mean_b = poptvar_k.loc[0,'b']  # Example mean of parameter
    SD_b = poptvar_k.loc[0,'bsd'] # Example standard deviation

## (14) MONTE CARLO SIMULATION #######################################
    num_simulations = 1000

# Store results
    F_lake_Y_k = [] # Refresh before each new lake.
# Run the Monte Carlo simulation
    for _ in range(num_simulations):
        
        # Generate a random parameter value
        a_rand = np.random.normal(loc=mean_a, scale=SD_a)
        b_rand = np.random.normal(loc=mean_b, scale=SD_b)
        
        # Calculate F
        dfTw_k_use['F_lake_pm2']  = (a_rand * b_rand ** (dfTw_k_use['Tw'] - 20)) # Yield flux in mmol m-2 d-1
        # Multiply with lake area and divide with 1000 to yeild units mol Lake-1 d-1
        L_area = Alake[k]
        dfTw_k_use['F_lake'] = dfTw_k_use['F_lake_pm2'] * L_area / 1000 # Division with 1000 yield mol from mmol.
        
        # Gather no of DOY, mean Tw and sum of F all DOYs => mol lake-1 yr-1 (of DOYS above threshold temperature = open water period)
        F_lke_Ysim = dfTw_k_use.resample('YE').agg({'DOY':'count', 'Tw':'mean', 'F_lake':'sum'}) 
        F_lke_Ysim['lake'] = k
        F_lke_Ysim['Y'] = F_lke_Ysim.index.year
        # Store result
        F_lake_Y_k.append(F_lke_Ysim)
    F_lake_Y.append(F_lake_Y_k)
    flat_F_lke_Y = [df for case_results in F_lake_Y for df in case_results] # Make list of df:s to df:s.
    dfMC_F_lke_Y = pd.concat(flat_F_lke_Y, ignore_index=True) # Make single combined df.
    dfMC_F_lke_Y.to_pickle("dfMC_F_lke_Y.pkl") # Save MC output as separate file

# Creating DataFrame with regression parameters for Observed CH4 = c + f*Predicted   WHOLE LAKE INTEGRATED FLUX
Br_DF = pd.concat(BrPar, ignore_index=True)
Br_DF.columns = (['c', 'f', 'R2', 'SEc', 'SEf', 'pc', 'pf', 'conflow_c', 'confhigh_c', 'conflow_f', 'confhigh_f', 'RMSE', 'lke'])
# c is constant; f is factor
Bdf_coef_all = pd.concat(dfvar, ignore_index=True)
Bdf_coef_all.columns = (['a', 'b', 'asd', 'bsd','lake'])
    
# Manage Monte Carlo output to generate mean, sd, and conf interval by lake and year.
MC_all = []
for l in range(len(lake)):
    MCl = dfMC_F_lke_Y.loc[dfMC_F_lke_Y['lake'] == l]

    MC_all_l = []
    for y in range(2016, 2023):  # 2023 is exclusive, so this covers 2016–2022
        MCly = MCl.loc[MCl['Y'] == y]
        DOY = MCly['DOY'].mean()
        Tw_mn = MCly['Tw'].mean()
        Tw_min = MCly['Tw'].min()
        Tw_max = MCly['Tw'].max()
        Fmin = MCly['F_lake'].min() #lake grand min
        Fmax = MCly['F_lake'].max()
        Fmd = MCly['F_lake'].median()
        q75, q25 = np.percentile(MCly['F_lake'], [75,25])
        Fiqr = q75 - q25
        Fmn = MCly['F_lake'].mean()
        Fsd = MCly['F_lake'].std()
        F_SE = stats.sem(MCly['F_lake'])  # Standard error of the mean
        # 95% confidence interval using t-distribution
        confidence = 0.95
        n = len(MCly['F_lake'])
        h = F_SE * stats.t.ppf((1 + confidence) / 2, df=n-1)  # Margin of error
        # Interval
        F_CI_low = Fmn - h
        F_CI_up = Fmn + h
        Fpm2_min = Fmin/Alake[l]
        Fpm2_max = Fmax/Alake[l]
        year = y
        lkenr = lake[l]
        MC_all_ly = [l, lkenr, year, DOY, Tw_mn, Tw_min, Tw_max, Fmin, Fmax, Fmd, Fiqr, Fmn, Fsd, F_SE, F_CI_low, F_CI_up, Fpm2_min, Fpm2_max]
        dfMC_all_ly = pd.DataFrame([MC_all_ly], columns=['lakeno', 'lake', 'year', 'DOY', 'Tw', 'Tw_min', 'Tw_max', 'Fmin', 'Fmax','Fmd', 'Fiqr', 'Fmn', 'Fsd', 'F_SE', 'F_CI_low', 'F_CI_up','Fpm2_min', 'Fpm2_max'])
        MC_all_l.append(dfMC_all_ly)
    MC_all.append(MC_all_l)
    flat_MC_all = [df for case_results in MC_all for df in case_results] # Make list of df:s to df:s.
    dfMC_all = pd.concat(flat_MC_all, ignore_index=True) # Make single combined df.

# Anomalies for DOY, Tw and F
dfMC_anop =dfMC_all[['lakeno', 'lake', 'year', 'DOY', 'Tw', 'Fmin', 'Fmax', 'Fmn', 'Fsd']]
dfMC_anop['Fmn_kg'] = dfMC_anop['Fmn']*16.04/1000 # from mol lake-1 yr-1 to kg CH4 lake-1 yr-1  
dfMC_anoapp = []
for m in range(len(lake)):
    dfMC_ano_m = dfMC_anop.loc[dfMC_anop['lakeno'] == m]
    mymnDOY = dfMC_ano_m['DOY'].mean() #multi-year mean DOY open water length by lake
    mysdDOY = dfMC_ano_m['DOY'].std()
    mymnTw = dfMC_ano_m['Tw'].mean() #multi-year mean Tw by lake
    mysdTw = dfMC_ano_m['Tw'].std()
    mymnF = dfMC_ano_m['Fmn_kg'].mean() #multi-year mean F by lake
    mysdF = dfMC_ano_m['Fmn_kg'].std()
    dfMC_ano_m['anoDOY'] = dfMC_ano_m['DOY'] - mymnDOY
    dfMC_ano_m['anoTw'] = dfMC_ano_m['Tw'] - mymnTw
    dfMC_ano_m['anoFmn_kg'] = dfMC_ano_m['Fmn_kg'] - mymnF
    dfMC_ano_m['anoFmn_%'] = dfMC_ano_m['anoFmn_kg'] / dfMC_ano_m['Fmn_kg'] * 100
    dfMC_ano_m['Tw*DOY'] = dfMC_ano_m['DOY'] * dfMC_ano_m['Tw']
    mymnTwDOY = dfMC_ano_m['Tw*DOY'].mean()
    dfMC_ano_m['anoTw*DOY'] = dfMC_ano_m['Tw*DOY'] - mymnTwDOY
    dfMC_ano_m['anoTw*DOY_%'] = dfMC_ano_m['anoTw*DOY'] / dfMC_ano_m['Tw*DOY'] * 100
    
    dfMC_anoapp.append(dfMC_ano_m)
dfMC_ano = pd.concat(dfMC_anoapp, ignore_index=True)

    






