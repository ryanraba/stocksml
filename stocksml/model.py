import sys
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
random.seed(11)
np.random.seed(13)

# Globals
#SYMBOL = 'ARNC'
SYMBOL = 'SPY'
SUPPORT = ['VIXM', 'BND']

#TRAIN_START = '1992-01-01'
TRAIN_START = '2015-01-01'
TRAIN_END = '2018-05-29' #time.strftime("%Y-%m-%d")
MAXLEN = 10
TARGET_WINDOW = 5


####################################################
## define functions that compute various indicators
## from basic inputs (high, low, open, close)
####################################################
def BarTrend(highs, lows, opens, closes):
    hldiffs = np.array(highs) - np.array(lows)
    codiffs = np.array(closes) - np.array(opens)
    ratios = np.zeros((len(hldiffs)), dtype=float)
    for ii in range(len(hldiffs)):
        if hldiffs[ii] > 0.0:
            ratios[ii] = codiffs[ii]/hldiffs[ii]
    return ratios

def Percent(values):
    perc = np.zeros((len(values)), dtype=float)
    for ii in range(1,len(values)):
        if values[ii-1] > 0:
            perc[ii] = (values[ii] - values[ii-1])/values[ii-1]
    return perc

def Breakout(highs, lows):
    diffs = list(np.array(highs) - np.array(lows))
    std = np.std(diffs)
    values = np.zeros((len(diffs)), dtype=int)
    for ii in range(1, len(diffs)):
        if diffs[ii] > 1*std:
            values[ii] = 1
            if highs[ii] < highs[ii-1]:
                values[ii] = -1
    return list(values)


  
#####################################################################
## Download stock data and compute indicator features
##
## returns data structure for regression
##
######################################################################
import pandas_datareader.data as web
from datetime import datetime as dtf
from sklearn.preprocessing import PolynomialFeatures

print('-- downloading data')

dt = web.DataReader(SYMBOL, 'iex',
                    dtf.strptime(TRAIN_START, '%Y-%m-%d'),
                    dtf.strptime(TRAIN_END, '%Y-%m-%d'))
#raw_dates = list(dt.index.get_level_values(1).strftime('%Y-%m-%d'))  #list(dt.index.strftime('%Y-%m-%d'))
#raw_data = dt.values

for ff in range(len(SUPPORT)):
  dt = web.DataReader(SUPPORT[ff], 'iex',
                      dtf.strptime(TRAIN_START, '%Y-%m-%d'),
                      dtf.strptime(TRAIN_END, '%Y-%m-%d'))
  support_dates = list(dt.index.get_level_values(1).strftime('%Y-%m-%d'))
  if len(np.setdiff1d(raw_dates, support_dates)) > 0: print("ERROR: date ranges from selected stocks dont match")

  support_data += [dt.values]

print('-- data retrieved')
print('-- support data size ' + str(len(support_data)))

N = 7
FEATURES = N + N*len(SUPPORT)

# ORDER: close, high, low, open, volume
    
# build desired set of indicators for the model
nn = 1.0
bdata = np.zeros((len(raw_data), FEATURES), dtype=float)
bdata[:,0] = np.power(Percent(raw_data[:,1]), nn) # high
bdata[:,1] = np.power(Percent(raw_data[:,2]), nn) # low
bdata[:,2] = np.power(Percent(raw_data[:,3]), nn) # open
bdata[:,3] = np.power(Percent(raw_data[:,0]), nn) # close
bdata[:,4] = np.power(raw_data[:,4], nn) # volume
bdata[:,5] = np.power(BarTrend(raw_data[:,1], raw_data[:,2], raw_data[:,3], raw_data[:,0]), nn)
bdata[:,6] = np.power(Breakout(raw_data[:,1], raw_data[:,2]), nn)

for ff in range(len(SUPPORT)):
  bdata[:,N*ff+N] = np.power(Percent(support_data[ff][:,1]), nn) # high
  bdata[:,N*ff+N+1] = np.power(Percent(support_data[ff][:,2]), nn) # low
  bdata[:,N*ff+N+2] = np.power(Percent(support_data[ff][:,3]), nn) # open
  bdata[:,N*ff+N+3] = np.power(Percent(support_data[ff][:,0]), nn) # close
  bdata[:,N*ff+N+4] = np.power(support_data[ff][:,4], nn) # volume
  bdata[:,N*ff+N+5] = np.power(BarTrend(support_data[ff][:,1], support_data[ff][:,2], support_data[ff][:,3], support_data[ff][:,0]), nn)
  bdata[:,N*ff+N+6] = np.power(Breakout(support_data[ff][:,1], support_data[ff][:,2]), nn)

bdata[np.isnan(bdata)] = 0.0

pdata = PolynomialFeatures(degree=1, interaction_only=True, 
                          include_bias=False).fit_transform(bdata)

pdata[np.isnan(pdata)] = 0.0

# mean normalization
for kk in range(0,len(pdata[0])):
    if np.std(pdata[:,kk]) > 0.0:
        pdata[:,kk] = (pdata[:,kk] - np.mean(pdata[:,kk]))/np.ptp(pdata[:,kk])

# vectorization
data = np.zeros((len(pdata), MAXLEN, len(pdata[0])), dtype=np.float)
for ii in range(len(pdata)):
    for jj in range(MAXLEN):
        if (ii-jj) > 0:
          data[ii,jj] = pdata[ii-jj]


target = np.zeros((len(raw_data)), dtype=int)
for ii in range(len(target)-1):
    peak = np.max(raw_data[ii+1:ii+TARGET_WINDOW+1, 1])
    if (raw_data[ii,1] > 0.0) and (peak/raw_data[ii,1] > 1.01): 
      target[ii] = 1

DATES = raw_dates

print(len(raw_data), len(target), data.shape)
