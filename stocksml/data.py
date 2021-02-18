#    Copyright (C) 2021  Ryan Raba
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
this module will be included in the api
"""

import pandas as pd
import os
import numpy as np


####################################################
## define functions that compute various indicators
## from basic inputs (high, low, open, close)
####################################################
def BarTrend(highs, lows, opens, closes):
    ratios = (closes - opens) / (highs - lows)
    ratios[np.isnan(ratios)] = np.nanmean(ratios)
    return ratios

def RSI(values, period=14):
    deltas = np.diff(values)
    seed = deltas[:period+1]
    up = seed[seed>=0].sum()/period
    down = -seed[seed<0].sum()/period
    if (down == 0): down = -0.01/period
    rs = up/down
    rsi = np.zeros_like(values)
    rsi[:period] = 1. - 1./(1.+rs)
    for ii in range(period, len(values)):
        delta = deltas[ii-1]
        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        if (down == 0): down = -0.01/period
        rs = up/down
        rsi[ii] = 1. - 1./(1.+rs)
    #rsi = list((np.array(rsi)*2.0)-1.0)
    return rsi

def Percent(values):
    perc = np.zeros((len(values)), dtype=float)
    perc[1:] = np.diff(values)/values[:-1]
    return perc

def SMA(values,period):
    weigths = np.repeat(1.0, period)/period
    smas = list(np.convolve(values, weigths, 'valid'))
    pad = [0.0 for _ in range(len(values) - len(smas))]
    return pad + smas

def EMA(values, period):
    period = float(period)
    a = np.array(values)
    for ii in range(1,len(values)):
        a = a[ii] * (2.0 / (period + 1.0)) + a[ii-1] * (1.0 - (2.0 / (period + 1.0)))
    return a

def MACD(values, slow=26, fast=12):
    emaslow = EMA(values, slow)
    emafast = EMA(values, fast)
    macd = emafast - emaslow
    signal = EMA(macd, 5)
    return macd, signal

def Stochastic(closes, highs, lows, period=14):
    so = [0.0 for _ in range(len(closes))]
    for ii in range(period,len(closes)):
        rng = max(ii-period, 0)
        [high, low] = [max(highs[rng:ii]), min(lows[rng:ii])]
        if (high-low) > 0:
            so[ii] = (closes[ii]-low)/(high-low)
    so = SMA(so,3)
    return so

def Breakout(highs, lows):
    diffs = list(np.array(highs) - np.array(lows))
    std = np.std(diffs)
    values = np.zeros((len(diffs)), dtype=int)
    for ii in range(1, len(diffs)):
        if diffs[ii] > 2*std:
            values[ii] = 1
            if highs[ii] < highs[ii-1]:
                values[ii] = -1
    return list(values)





#####################################################################
##
##
##
######################################################################
def FetchData(symbols, apikey, start=None, stop=None, path=None, append=True):
    """
    Download symbol data from iex using provided api key, counts against quota

    Parameters
    ----------
    symbols : list of str
        list of ticker symbols to retrieve
    apikey : str
        api token of the iex account to use
    start : str
        start date of historical prices to retrieve. Format is yyyy-mm-dd.
        Default None uses current date
    stop : str
        stop date of historical prices to retrieve. Format is yyyy-mm-dd.
        Default None uses current date
    path : str
        path of folder to place downloaded data. Default None uses current directory
    append : bool
        append new data to existing file or create if missing. Duplicate dates ignored.
        False will overwrite file. Default True
    """
    import time
    import pandas_datareader.data as web

    if start is None: start = time.strftime("%Y-%m-%d")
    if stop is None: stop = time.strftime("%Y-%m-%d")
    if path is None: path = './'
    if not path.endswith('/'): path = path + '/'
    
    sdf = pd.DataFrame([])
    for symbol in symbols:
        print('fetching %s data...' % symbol, end=' ')
        ddf = web.DataReader(symbol, 'iex', start, stop, api_key=apikey)
        print('%s days' % str(len(ddf)))

        if append and os.path.exists(path+symbol+'.csv'):
            print('merging with existing file for %s...' % symbol)
            pdf = pd.read_csv(path+symbol+'.csv').set_index('date')
            new_dates = np.setdiff1d(pdf.index.values, ddf.index.values)
            ddf = pd.concat([ddf, pdf.loc[new_dates]]).sort_index()
            
        ddf.to_csv(path+symbol+'.csv')
        
    return



##############################################################
def LoadData(symbols=None, path=None):
    """
    Load price data from CSV files
    
    Parameters
    ----------
    symbols : list of str
        list of ticker symbol files to load. Files should be in the form of symbol.csv.
        Default None loads all files in provided directory.
    path : str
        path to symbol data files. Default None uses included demonstration data folder location

    Returns
    -------
    pandas.DataFrame
        symbol dataframe
    """
    import pkg_resources

    if path is None:
        path = pkg_resources.resource_filename('stocksml', 'data/')
    if not path.endswith('/'): path = path + '/'

    sdf = None
    if os.path.exists(path):
        if symbols is None:
            symbols = [ff.split('.csv')[0] for ff in os.listdir(path)]
    
        for symbol in symbols:
            if not os.path.exists(path+symbol+'.csv'): continue
            
            pdf = pd.read_csv(path+symbol+'.csv').set_index('date')
            pdf = pdf.rename(columns=dict([(cc, symbol.lower() + '_' + cc) for cc in pdf.columns]))
            sdf = pdf if sdf is None else sdf.merge(pdf, 'inner', left_index=True, right_index=True)

    symbols = list(np.unique([ss.split('_')[0] for ss in sdf.columns]))
    return sdf, symbols


################################################################
def BuildData(sdf):
    """
    Transform price data from symbol dataframe to training feature set

    Parameters
    ----------
    sdf : pandas.DataFrame
        symbol dataframe
    
    Returns
    -------
    pandas.DataFrame
        feature dataframe
    """
    symbols = list(np.unique([ss.split('_')[0] for ss in sdf.columns]))
    cpdf = pd.DataFrame()
    for ii, ss in enumerate(symbols):
        print('building %s data...' % ss.upper())
        pdf = sdf[[cc for cc in sdf.columns if cc.startswith(ss)]]
        
        # build features
        dm = [Percent(pdf[ss+'_high'].values), Percent(pdf[ss+'_low'].values), Percent(pdf[ss+'_open'].values), Percent(pdf[ss+'_close'].values)]
        dm += [BarTrend(pdf[ss+'_high'].values, pdf[ss+'_low'].values, pdf[ss+'_open'].values, pdf[ss+'_close'].values)]
        dm = np.array(dm).transpose()
    
        # mean normalization
        for kk in range(0, dm.shape[1]):
            if np.std(dm[:, kk]) > 0.0:
                dm[:, kk] = (dm[:, kk] - np.mean(dm[:, kk])) / np.ptp(dm[:, kk])

        # make a dataframe out of the features and merge to combined dataframe
        fpdf = pd.DataFrame(dm, index=pdf.index, columns=[ss + str(ff) for ff in range(dm.shape[1])])
        cpdf = cpdf.merge(fpdf, 'right', left_index=True, right_index=True)
        
    return cpdf



