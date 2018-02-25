#############################################################################
##
## Stock Prediction Regression
## 
##
## Prereqs:
##   numpy
##   scikit-learn (pip install scikit-learn)
##   pandas (pip install pandas)
##   pandas-datareader (pip install pandas-datareader)
##   matplotlib (sudo apt-get install python-matplotlib)
##   h5py (pip install h5py)
##   keras
##
#############################################################################
from __future__ import print_function
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

#TRAIN_START = '1992-01-01'
TRAIN_START = '2010-01-01'
TRAIN_END = time.strftime("%Y-%m-%d")
TDATES = []
TDATA = []
TTARGET = []
MODE = 0
MAXLEN = 5

MARKET = {}

run_options = str(sys.argv[1:])
if ('help' in run_options) or ('-h' in run_options) or (len(sys.argv[1:]) == 0):
    print('\nUsage:')
    print('  python stock_c.py simulate')
    print('  python stock_c.py predict [yyyy-mm-dd]')
    print('  python stock_c.py visualize_data [0-12]\n')


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
    for ii in range(1,len(values)):
        if values[ii-1] > 0:
            perc[ii] = (values[ii] - values[ii-1])/values[ii-1]
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
        a[ii] = a[ii] * (2.0 / (period + 1.0)) + a[ii-1] * (1.0 - (2.0 / (period + 1.0)))
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
## Download stock data and compute indicator features
##
## returns data structure for regression
##
######################################################################
def BuildData():
    import pandas_datareader.data as web
    from datetime import datetime as dtf
    from sklearn.preprocessing import PolynomialFeatures

    print('-- building data')

    dt = web.DataReader(SYMBOL, 'google',
                        dtf.strptime(TRAIN_START, '%Y-%m-%d'),
                        dtf.strptime(TRAIN_END, '%Y-%m-%d'))
    raw_dates = list(dt.index.strftime('%Y-%m-%d'))
    raw_data = dt.values
    for date in raw_dates:
      MARKET[date] = list(dt.ix[date].values)
	
	
    #raw_dates = sorted(list(MARKET.keys()))
    #raw_data = np.zeros((len(raw_dates), 5), dtype=float)
    #for ii, date in enumerate(raw_dates):
    #    raw_data[ii] = MARKET[date]

    # build desired set of indicators for the model
    nn = 1.0
    bdata = np.zeros((len(raw_data), 5), dtype=float)
    #bdata[:,0] = np.power(raw_data[:,1], nn) # high
    #bdata[:,1] = np.power(raw_data[:,2], nn) # low
    #bdata[:,2] = np.power(raw_data[:,0], nn) # open
    #bdata[:,3] = np.power(raw_data[:,3], nn) # close
    bdata[:,0] = np.power(Percent(raw_data[:,1]), nn) # high
    bdata[:,1] = np.power(Percent(raw_data[:,2]), nn) # low
    bdata[:,2] = np.power(Percent(raw_data[:,0]), nn) # open
    bdata[:,3] = np.power(Percent(raw_data[:,3]), nn) # close
    bdata[:,4] = np.power((raw_data[:,0]-raw_data[:,3])/raw_data[:,0], nn) # close
    #bdata[:,4] = np.power(raw_data[:,4], nn) # volume
    #bdata[:,5] = np.power(Percent(raw_data[:,1] - SMA(raw_data[:,1], 5)), nn)
    #bdata[:,6] = np.power(raw_data[:,1] - SMA(raw_data[:,1], 3), nn)
    #bdata[:,7] = np.power(raw_data[:,1] - SMA(raw_data[:,1], 10), nn)
    #bdata[:,8] = np.power(Percent(raw_data[:,1]), nn)
    #bdata[:,9] = np.power(RSI(raw_data[:,1], 5), nn)
    #bdata[:,10], bdata[:,11] = MACD(raw_data[:,1], 10, 2)
    #bdata[:,10], bdata[:,11] = [np.power(bdata[:,4], nn), np.power(bdata[:,5], nn)]
    #bdata[:,12] = np.power(EMA(Stochastic(raw_data[:,3], raw_data[:,1], raw_data[:,2], 6), 3), nn)
    #bdata[:,13] = np.power(Percent(raw_data[:,4] - SMA(raw_data[:,4],5)), nn)
    #bdata[:,14] = np.power(Breakout(raw_data[:,1], raw_data[:,2]), nn)
    #bdata[:,15] = np.power(BarTrend(raw_data[:,1], raw_data[:,2], raw_data[:,0], raw_data[:,3]), nn)

    bdata[np.isnan(bdata)] = 0.0

    data = PolynomialFeatures(degree=1, interaction_only=False, 
                              include_bias=False).fit_transform(bdata)

    data[np.isnan(data)] = 0.0

    # mean normalization
    for kk in range(0,len(data[0])):
        if np.std(data[:,kk]) > 0.0:
            data[:,kk] = (data[:,kk] - np.mean(data[:,kk]))/np.ptp(data[:,kk])

    target = np.zeros((len(raw_data)), dtype=float)
    for ii in range(len(target)-3):
        high_avg = (raw_data[ii+1,1] + raw_data[ii+2,1] + raw_data[ii+3,1])/3.0 
        high_to_high = high_avg - raw_data[ii,1]
        if (raw_data[ii,1] > 0.0): target[ii] = high_to_high/raw_data[ii,1]
        
    target[np.isnan(target)] = 0.0
        
    return raw_dates, data, target



#############################################################################
## take model input structure and convert to 3-dim float array.
## 
## Output: X (model input) and y (model output/truth) 3D data structures
#############################################################################
def Vectorize(dateidxs, cleanse=False):
    X = np.zeros((len(dateidxs), MAXLEN, len(TDATA[0])), dtype=np.float)
    y = np.zeros((len(dateidxs)), dtype=np.float)
    for ii in range(len(dateidxs)):
        for jj in range(MAXLEN):
            X[ii,jj] = TDATA[dateidxs[ii]-jj]
        y[ii] = TTARGET[dateidxs[ii]]

    return (X, y)
   


#######################################################################################
## Train model up to specific day and then predict action for that day
##
## Output: prediction rough value and truth if available
#######################################################################################
def PredictDate(date, model=None, thresh=0.0):
    from keras.models import Model
    from keras.layers import Input, Dense, SimpleRNN, LSTM
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras import regularizers as reg
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    didx = len(TDATES)
    if date in TDATES:
        didx = TDATES.index(date)
    
    if model == None:
        print('-- building model')
        X, y = Vectorize(range(didx))

        ystd = np.std(y)
        yc = np.zeros((len(y), 4), dtype=int)
        yi = np.zeros((len(y)), dtype=int)
        pp = -99.0
        for ii in range(3):
            mm = ii-1.
            yc[(y > pp*ystd) & (y < mm*ystd),ii] = 1
            yi[(y > pp*ystd) & (y < mm*ystd)] = ii
            pp = mm
        yc[y > ystd, 3] = 1
        yi[y > ystd] = 3
        
        yb = np.array(y > 0, dtype=int)

        ins = Input(shape=(len(X[0]),len(X[0,0])))
        hh = LSTM(256, activation='relu', kernel_regularizer=reg.l1(1e-5),
                  return_sequences=True)(ins)
        #hh = SimpleRNN(32, activation='tanh', kernel_regularizer=reg.l1(3e-5),
        #               return_sequences=True)(hh)
        #hh = SimpleRNN(32, activation='tanh', kernel_regularizer=reg.l1(3e-5),
        #               return_sequences=True)(hh)
        #hh = SimpleRNN(32, activation='tanh', kernel_regularizer=reg.l1(3e-5),
        #               return_sequences=True)(hh)
        hh = LSTM(256, activation='relu', kernel_regularizer=reg.l1(1e-5))(hh)
        #hh = Dense(512, activation='tanh', kernel_regularizer=reg.l1(1e-4))(hh)
        outs = Dense(4, activation='softmax')(hh)
        
        model = Model(inputs=ins, outputs=outs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        mweights = model.get_weights()

        fig, ax = plt.subplots()

        if MODE == 0: X = X.reshape(-1, MAXLEN*len(TDATA[0]))
        
        vidxs = random.sample(range(len(X)), int(0.15*len(X)))
        tidxs = list(np.setdiff1d(range(len(X)), vidxs))
        Xt, Xv = [X[tidxs], X[vidxs]]
        ytc, yvc = [yc[tidxs], yc[vidxs]]
        yti, yvi = [yi[tidxs], yi[vidxs]]
        ytb, yvb = [yb[tidxs], yb[vidxs]]
        
        mm = len(Xt)
        for ii in range(10,11):
            #print('training iteration ' + str(ii))
            Xts, ytcs = [Xt[:int(0.1*ii*mm)], ytc[:int(0.1*ii*mm)]]
            ytis, ytbs = [yti[:int(0.1*ii*mm)], ytb[:int(0.1*ii*mm)]]

            if MODE == 0:
                #model = LogisticRegression(penalty='l1', C=4.0, fit_intercept=True, max_iter=4000, 
                #                           random_state=17, verbose=0, n_jobs=-1).fit(Xts, ytbs) 
                model = SVC(C=1600.0, kernel='rbf', probability=True, shrinking=True, 
                            verbose=False, max_iter=-1, random_state=17,
                            decision_function_shape='ovr').fit(Xts, ytbs)

            elif MODE == 1:
                model.set_weights(mweights)
                model.fit([Xts], [ytcs], epochs=1000, batch_size=512, shuffle=True,
                          validation_data=([Xv], [yvc]), verbose=1,
                          callbacks=[ModelCheckpoint('stock_nn.hf5','val_loss',0, True,True),
                                     EarlyStopping(monitor='val_loss', patience=10)])
            
            if MODE==1: preds = model.predict(Xts)
            elif MODE==0: preds = model.predict_proba(Xts)[:,1]
            if isinstance(preds[0], np.ndarray):
                preds = np.sum(preds[:,2:4],1) - np.sum(preds[:,:2],1) + 0.5
            preds = np.array(preds > 0.5, dtype=int)
            err = 1.0 - accuracy_score(ytbs.reshape(len(ytbs),-1), preds)
            ax.scatter( [ii], [err], [100], color='b')

            if MODE==1: preds = model.predict(Xv)
            elif MODE==0: preds = model.predict_proba(Xv)[:,1]
            if isinstance(preds[0], np.ndarray):
                preds = np.sum(preds[:,2:4],1) - np.sum(preds[:,:2],1) + 0.5
            preds = np.array(preds > 0.5, dtype=int)
            err = 1.0 - accuracy_score(yvb.reshape(len(yvb),-1), preds)
            ax.scatter( [ii], [err], [100], color='m')
        
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,y2))
        plt.show(block=False)

        if MODE==1: preds = model.predict(X)
        elif MODE==0: preds = model.predict_proba(X)[:,1]
        if isinstance(preds[0], np.ndarray):
            preds = np.sum(preds[:,2:4],1) - np.sum(preds[:,:2],1) + 0.5
        thresh = np.mean(preds) + 0.8*np.std(preds)
        
    if MODE == 1: model.load_weights('stock_nn.hf5')

    X, y = Vectorize([didx-1])
    if MODE == 0:
      X = X.reshape(-1, MAXLEN*len(TDATA[0]))
      preds = model.predict_proba(X)[-1,1]
    if MODE==1:
      preds = model.predict(X)[-1]
    if isinstance(preds, np.ndarray): 
      preds = np.sum(preds[2:4]) - np.sum(preds[:2]) + 0.5

    guess = int(preds > 0.5)
    buy = int(preds > thresh)

    prev_high = MARKET[TDATES[didx-1]][1] #TDATA[didx-1,0]

    if buy == 1:
        limit = prev_high
    else:
        limit = prev_high #  * (1.0 + preds[-1])
        
    return guess, buy, preds, y[-1], limit, model, thresh



#######################################################################################
## Back test model of over given date range by simulating performance against a 
## specific trading strategy. This trading strategy buys on days where prices are
## expected to rise and sells on days they are expected to decline.
##
## The simulation date range must not overlap with the data used to train the model or
## results will be artificially inflated
##
## Output: returns performance metrics as:
##   gross earnings
##   baseline stock performance using buy and hold
##   percentage of positive days successfully exploited
##   percentage of buys that were the correct call
##   overall accuracy in predicting buys and sells
##   R score of model
########################################################################################
def SimulateDates(start_date, end_date, verbose):
    
    dates = sorted(list(MARKET.keys()))
    dates = dates[dates.index(start_date):dates.index(end_date)+1]
    
    cash = 1000.0
    y_truth, y_preds = [[], []]
    tcnt, prev_high, prev_low, prev_close, shares, target = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    model, thresh = [None, 0.0]
    pt = 0.5 # threshold for positive response

    for ii, date in enumerate(dates):
        open_val = MARKET[date][0]
        high_val = MARKET[date][1]
        low_val = MARKET[date][2]
        close_val = MARKET[date][3]
        if ii==0: initial_shares = cash/open_val
        if ii==0: prev_high = open_val
        
        guess, buy, pval, tval, limit, model, thresh = PredictDate(date, model, thresh)
        y_preds += [pval]
        y_truth += [tval]
        starting_cash = cash

        # buy
        if (shares == 0.0) and (buy == 1):
            shares, cash, tcnt = [cash / open_val, 0.0, tcnt + 1]
            #limit = open_val * (1.0 + pval + 0.003)
            #if (high_val >= limit) and (limit > 0.0):
            #    cash, shares = [shares * limit, 0.0]
            #if open_val <= limit:
            #    shares, cash, tcnt = [cash / open_val, 0.0, tcnt + 1]
            #elif low_val <= limit:
            #    shares, cash, tcnt = [cash / limit, 0.0, tcnt + 1]
        
        # sell
        elif (shares > 0.0) and (guess == 0):
            if open_val > limit:
                cash, shares = [shares * open_val, 0.0]
            elif high_val >= limit:
                cash, shares = [shares * limit, 0.0]

        if verbose: 
            print(date,"{:3.2f} / {:3.2f}".format(open_val,high_val),
                  int(starting_cash),guess,int(tval > 0.0),int(cash), pval, tval)

        prev_high = high_val


    # final metrics
    cash += shares * close_val
    earnings = str(int(cash))
    baseline = str(int(initial_shares*close_val))
    
    y_preds, y_truth = [np.array(y_preds), np.array(y_truth)]
    actual_pos = float(np.sum(y_truth > 0.0)) / len(y_truth)
    true_pos = float(np.sum((y_preds > pt) & (y_truth > 0.0)))
    accuracy = float(np.sum( (y_preds > pt) == (y_truth > 0.0) )) / len(y_preds)
    precision =  true_pos / (true_pos + np.sum((y_preds > pt) & (y_truth < 0.0)))
    recall = true_pos / (true_pos + np.sum((y_preds < pt) & (y_truth > 0.0)))
    
    actual_pos = "{:4.2f}".format(actual_pos)
    accuracy = "{:4.2f}".format(accuracy)
    precision = "{:4.2f}".format(precision)
    recall = "{:4.2f}".format(recall)
    if verbose:
        print("====================")
        print("Model: $" + earnings + "  Baseline: $" + baseline)
        print("Precision: " + precision)
        print("Recall: " + recall)
        print("Accuracy: " + accuracy)
        print("Actual Positive: " + actual_pos)

    return earnings, baseline, precision, recall, accuracy, actual_pos, tcnt



#################################################################################
#################################################################################
#################################################################################
#################################################################################
## Main Program Body                                                           ##
#################################################################################
#################################################################################
#################################################################################
#################################################################################

TDATES, TDATA, TTARGET = BuildData()

if 'simulate' in run_options:
    datelist = [#['2016-07-01', '2016-07-29'],
                #['2016-08-01', '2016-08-31'],
                #['2016-09-01', '2016-09-30'],
                #['2016-10-03', '2016-10-31'],
                #['2016-11-01', '2016-11-30'],
                #['2016-12-01', '2016-12-30'],
                #['2017-01-03', '2017-01-31'],
                #['2017-02-01', '2017-02-28'],
                #['2017-03-01', '2017-03-31'],
                #['2017-04-03', '2017-04-28'],
                ['2016-07-01', '2016-12-30']]
                #['2017-01-03', '2017-04-28']]

    for dates in datelist:
        results = SimulateDates(dates[0], dates[1], False) #len(datelist) == 1)
        print(dates, results)
    plt.show(block=True)
    exit()

elif 'predict' in run_options:
    date = sys.argv[len(sys.argv)-1]
    results = PredictDate(date)
    print(date, results[:-2])
    exit()


elif 'visualize_data' in run_options:
    import random as rn

    feat = int(sys.argv[len(sys.argv)-1])

    X, y = Vectorize(range(len(TDATES)))
    idxs = rn.sample(range(20,len(X)), 50)

    fig = plt.figure()    
    pplts = [fig.add_subplot(261), fig.add_subplot(262), fig.add_subplot(263),
             fig.add_subplot(264), fig.add_subplot(265), fig.add_subplot(266)]
    nplts = [fig.add_subplot(267), fig.add_subplot(268), fig.add_subplot(269),
             fig.add_subplot(2,6,10), fig.add_subplot(2,6,11), fig.add_subplot(2,6,12)]

    pcnt, ncnt = [0, 0]
    for idx in idxs:
        if (y[idx] > 0.0) and (pcnt < 6):
            pplts[pcnt].plot(list(X[idx-20:idx,feat]),'bo-')
            pcnt += 1
        elif (y[idx] <= 0.0) and (ncnt < 6):
            nplts[ncnt].plot(list(X[idx-20:idx,feat]), 'ro-')
            ncnt += 1
    plt.show()
    exit()



#with open('tickerlist.txt', 'r', 1) as fid:
#    data = fid.read().splitlines()    
#symbol_list = data[0].split(', ')
#good_symbols = []

symbol_list = ['AES', 'AMT', 'AVGO', 'CA', 'CMS', 'COH', 'ED', 'DTE', 'ETR', 'EFX', 'ES', 'FE', 'FTR', 'GGP', 'HRL', 'HUM', 'IFF', 'IRM', 'KLAC', 'MAC', 'MNK', 'MAS', 'NI', 'PCG', 'PNW']

returns = []
baselines = []
for symbol in symbol_list:
    SYMBOL = symbol
    TDATES, TDATA, TTARGET = BuildData()
    results = SimulateDates('2016-07-01', '2016-12-30', False)
#    #if (float(results[-1]) > 0.4): 
    print(symbol, results)
    returns.append(float(results[0]))
    baselines.append(float(results[1]))
#    #    good_symbols.append(symbol)

print(str(np.mean(returns)), str(np.mean(baselines)))
#print(good_symbols)
#exit()
