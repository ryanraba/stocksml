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

import pandas_datareader.data as web
import pandas as pd
from datetime import datetime as dtf
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
## Download stock data from iex using api key from text file
##
## will count against monthly quota
######################################################################
def FetchData(symbols, train_start, train_end, append=False):
    cpdf = pd.DataFrame([])
    for symbol in symbols:
        if append or not os.path.exists('./data/%s.csv' % symbol):
            with open('iex_key.txt', 'r') as fid:
                apikey = fid.read().strip()

            print('fetching %s data...' % symbol)
            pdf = web.DataReader(symbol, 'iex', dtf.strptime(train_start, '%Y-%m-%d'), dtf.strptime(train_end, '%Y-%m-%d'), api_key=apikey)
    
            if append and os.path.exists('./data/%s.csv' % symbol):
                opdf = pd.read_csv('./data/%s.csv' % symbol).set_index('date')
                pdf = pd.concat([opdf, pdf]).sort_index()
    
            pdf.to_csv('./%s.csv' % symbol)
            
        pdf = pd.read_csv('./data/%s.csv' % symbol).set_index('date')
        pdf = pdf.rename(columns=dict([(cc,symbol.lower()+'_'+cc) for cc in pdf.columns]))
        cpdf = cpdf.merge(pdf, 'right', left_index=True, right_index=True)
        
    return cpdf



################################################################
## read CSV files in to Pandas Dataframe and build features
##
################################################################
def BuildData(symbols):
    cpdf = pd.DataFrame([])
    for ii, symbol in enumerate(symbols):
        print('building %s data...' % symbol)
        pdf = pd.read_csv('./data/%s.csv' % symbol).set_index('date')
    
        # build features
        dm = [Percent(pdf.high.values), Percent(pdf.low.values), Percent(pdf.open.values), Percent(pdf.close.values)]
        dm += [BarTrend(pdf.high.values, pdf.low.values, pdf.open.values, pdf.close.values)]
        dm = np.array(dm).transpose()
    
        # mean normalization
        for kk in range(0, dm.shape[1]):
            if np.std(dm[:, kk]) > 0.0:
                dm[:, kk] = (dm[:, kk] - np.mean(dm[:, kk])) / np.ptp(dm[:, kk])

        # make a dataframe out of the features and merge to combined dataframe
        fpdf = pd.DataFrame(dm, index=pdf.index, columns=[symbol.lower() + str(ff) for ff in range(dm.shape[1])])
        cpdf = cpdf.merge(fpdf, 'right', left_index=True, right_index=True)
        
    return cpdf



################################################################
## add third dimension for rnn/cnn based models
##
################################################################
def Vectorize(values, depth=1):
    dx = np.repeat(values[:, None, :], depth, axis=1)
    for ii in range(1,depth):
        dx[:,ii,:] = np.vstack((np.zeros((ii,values.shape[1])), dx[:-ii,ii,:]))
    
    return dx



