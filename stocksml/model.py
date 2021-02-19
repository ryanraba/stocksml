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

import os
import numpy as np
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
random.seed(11)
np.random.seed(13)



###################################################
def BuildModel(fdf, choices, layers=[('rnn',32),('dnn',64),('dnn',32)], depth=5, count=2):
    """
    Build a model with the given structure

    Parameters
    ----------
    fdf : pandas.DataFrame
        feature dataframe
    choices : int
        number of ticker symbols model can choose between
    layers : list of tuples
        list of tuples defining structure of model. Each tuple is (layer, size) where layer can
        be 'dnn', 'cnn', 'lstm', 'rnn', or 'drop'. Default is a 3-layer model with [('rnn',32),('dnn',64),('dnn',32)]
    depth : int
        depth of time dimension for recurrent and convolutional networks (rnn, cnn, lstm). Ignored if using dnn only.
        Default is 5.
    count : int
        number of models to build. Default is 2
        
    Returns
    -------
    list of keras.Model, numpy.ndarray
        list of keras Models built, compiled and ready for training along with the appropriate data array for training
    """
    from keras.models import Model, clone_model
    from keras.layers import Input, Dense, SimpleRNN, LSTM, Conv1D, Flatten, Dropout
    
    # check to see if depth is needed
    layer_types = np.unique([nn[0] for nn in layers])
    if ('cnn' in layer_types) or ('rnn' in layer_types) or ('lstm' in layer_types):
        dx = np.repeat(fdf.values[:, None, :], depth, axis=1)
        for ii in range(1, depth):
            dx[:, ii, :] = np.vstack((np.zeros((ii, fdf.values.shape[1])), dx[:-ii, ii, :]))
    else:
        dx = fdf.values
        
    # fixed input to model
    ins = Input(shape=dx.shape[1:], name='input')
    hh = ins
    
    # build middle layers according to specified structure
    for ll, layer in enumerate(layers):
        name = layer[0]+'_'+str(ll)
        flatten = (ll>=len(layers)-1) or (layers[ll+1][0]=='dnn') or ((ll<len(layers)-2) and (layers[ll+1][0]=='drop') and (layers[ll+2][0]=='dnn'))
        if layer[0] is 'dnn':
            hh = Dense(layer[1], activation='tanh', name=name)(hh)
        elif layer[0] is 'rnn':
            hh = SimpleRNN(layer[1], activation='tanh', return_sequences=not flatten, name=name)(hh)
        elif layer[0] is 'lstm':
            hh = LSTM(layer[1], activation='tanh', return_sequences=not flatten, name=name)(hh)
        elif layer[0] is 'cnn':
            hh = Conv1D(layer[1], 3, padding='valid', activation='relu', name=name)(hh)
            if flatten:
                hh = Flatten(name='flatten')(hh)
        elif layer[0] is 'drop':
            hh = Dropout(layer[1], name=name)(hh)
    
    # fixed outputs from model
    action = Dense(5, activation='softmax', name='action')(hh)
    symbol = Dense(choices, activation='softmax', name='symbol')(hh)
    limit = Dense(1, activation='tanh', name='limit')(hh)

    model = Model(inputs=ins, outputs=[action, symbol, limit], name='model')
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy', 'mse'], optimizer='adam')

    models = [model]
    for mm in range(count-1):
        models += [clone_model(model)]
        models[-1].compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mse'], optimizer='adam')

    return models, dx




###################################################
def LearnStrategy(models, sdf, dx, symbols, baseline=None, days=5, maxiter=1000, notebook=False):
    """
    Learn a trading strategy by training models against provided data

    Parameters
    ----------
    models : list of keras.Model
        list of prebuilt models to train
    sdf : pandas.DataFrame
        symbol dataframe with price information
    dx : numpy.array
        vectorized training data
    symbols : list of str
        list of ticker symbols available to the trading strategy. Must all be contained in sdf
    baseline : str
        ticker symbol to use for baselining of trading strategy.
        Default None performs no baseline
    days : int
        number of days to use for trading strategy. Default is 5
    maxiter : int
        maximum number of training iterations. Default is 1000
    notebook : bool
        configures live plots for running in a Jupyter notebook.  Default is False
    """
    import matplotlib.pyplot as plt
    from stocksml import EvaluateChoices
    if notebook:
        from IPython import display
    
    fig, ax = plt.subplots(2, 2, figsize=(16,8))
    cum_results = np.zeros((maxiter,4))
    cum_choices = np.zeros((maxiter,len(models)))
    cum_symbols = np.zeros((maxiter,len(models)))
    cum_limits = np.zeros((maxiter, len(models)))

    for ee in range(maxiter):
        
        # randomly select a week of data
        ss = np.random.randint(10,dx.shape[0]-6)
        dates = list(sdf.index.values[ss:ss + days])
        
        # each model makes a set of trades for the week
        results = np.zeros((len(models)))
        choices = [[]] * len(models)
        for mm in range(len(models)):
            preds = models[mm].predict_on_batch(dx[ss:ss+days])
            # list of tuples (action, symbol, limit)
            choices[mm] = [(np.argmax(preds[0][dd]), np.argmax(preds[1][dd]), preds[2][dd][0]) for dd in range(days)]
            
            # evaluate the performance of the trade choices
            # normalize results to the baseline buy/hold strategy
            results[mm], reference, log = EvaluateChoices(sdf, symbols, dates, choices[mm], baseline)
            if reference > 0: results[mm] = results[mm]/reference

        # the model earning the most money wins
        # the winner defines the truth data for this week
        # if neither is successful, skip training this week
        winner = np.argmax(results)
        cum_results[ee][0] = results[winner]
        cum_choices[ee] = np.std([cc[0] for cc in choices[mm] for mm in range(len(choices))], axis=0)
        cum_symbols[ee] = np.std([cc[1] for cc in choices[mm] for mm in range(len(choices))], axis=0)
        cum_limits[ee] = np.std([cc[2] for cc in choices[mm] for mm in range(len(choices))], axis=0)
        if np.max(results) <= 1.0: continue
        if np.max(np.abs(np.diff(results))) < 0.0025: continue

        truth = [np.zeros((days,5), dtype=int), np.zeros((days,len(symbols)), dtype=int), []]
        for dd in range(days): truth[0][dd,choices[winner][dd][0]] = 1
        for dd in range(days): truth[1][dd, choices[winner][dd][1]] = 1
        truth[2] = np.array([ll[2] for ll in choices[winner]]).reshape(-1, 1).clip(-0.05, 0.05)

        # train losing model with winners truth data if winner made money
        for mm in range(len(models)):
            if mm == winner: continue
            cum_results[ee][1:] = models[mm].train_on_batch(dx[ss:ss+days], truth)[1:]
            #print('updated model %s'%str(mm), cum_results[ee][1:], results)
            
        # update the plots
        for mm in range(2): ax[mm,0].clear(), ax[mm,1].clear()
        #for mm in range(len(models)):
        #    rc = ax[0, 0].scatter(np.arange(days), [cc[0] for cc in choices[mm]])
        #    rc = ax[1, 0].scatter(np.arange(days), [cc[1] for cc in choices[mm]])
        train_points = np.where(cum_results[:, 1] > 0)[0]
        rc = ax[0, 0].plot(np.arange(ee), cum_choices[:ee, 0], marker='.', linewidth=0.0)
        rc = ax[0, 0].plot(np.arange(ee), cum_symbols[:ee, 0], marker='.', linewidth=0.0)
        rc = ax[1, 0].plot(np.arange(ee), cum_limits[:ee, 0], marker='.', linewidth=0.0)
        rc = ax[0, 1].plot(np.arange(ee), cum_results[:ee, 0])
        rc = ax[1, 1].plot(train_points, cum_results[train_points, 1:])
        ax[1,1].set_yscale('log'), ax[1,0].set_xlabel('Trading Iteration'), ax[1,1].set_xlabel('Training Iteration')
        ax[0,0].set_ylabel('Model Choice Standard Deviation'), ax[1,0].set_ylabel('Model Limit Standard Deviation')
        ax[0,1].set_ylabel('Normalized Performance'), ax[1,1].set_ylabel('Model Training Loss')
    
        if notebook:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        else:
            plt.pause(0.05)
    
    if notebook: plt.close()



###################################################
def ExamineStrategy(model, sdf, dx, symbols, start_date, days=5, baseline=None):
    """
    Explore a strategy learned by a model
    
    Parameters
    ----------
    model : keras.Model
        trained model to execute strategy with
    sdf : pandas.DataFrame
        symbol dataframe with price information
    dx : numpy.array
        vectorized training data
    symbols : list of str
        list of ticker symbols available to the trading strategy. Must all be contained in sdf
    start_date : str
        date to start trading strategy on. yyyy-mm-dd format
    days : int
        number of days to run strategy for. Default is 5
    baseline : str
        ticker symbol to use for baselining of trading strategy.
        Default None performs no baseline
    """
    from stocksml import EvaluateChoices

    start_index = sdf.index.get_loc(start_date) - 1  # strategy executes on the next day
    dates = list(sdf.index.values[start_index:start_index+days])
    
    preds = model.predict_on_batch(dx[start_index:start_index+days])
    
    # list of tuples (action, symbol, limit)
    choices = [(np.argmax(preds[0][dd]), np.argmax(preds[1][dd]), preds[2][dd][0]) for dd in range(days)]
    
    # evaluate the performance of the trade choices
    results, reference, log = EvaluateChoices(sdf, symbols, dates, choices, baseline)
    print(log)



#########################################################
def Demo(notebook=False):
    """
    Demonstration of how to use this package
    
    Parameters
    ----------
    notebook : bool
        set live plots for running properly in Jupyter notebooks.  Default is False
    """
    from stocksml import LoadData, BuildData

    # retrieve symbol dataframe
    sdf, symbols = LoadData()

    # build to feature dataframe
    fdf = BuildData(sdf)

    models, dx = BuildModel(fdf, len(symbols), count=2)

    LearnStrategy(models, sdf, dx, symbols, 'SPY', 5, 1000, notebook)

    ExamineStrategy(models[0], sdf, dx, symbols, '2021-02-01', days=5, baseline='SPY')
