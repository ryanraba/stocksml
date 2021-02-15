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

import os
import numpy as np
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
random.seed(11)
np.random.seed(13)



###################################################
## Build a model with the given structure
##
## layers = list of tuples defining structure of model
##          each tuple is (layer, size) where layer can
#           be 'dnn', 'cnn', 'lstm', 'rnn', or 'drop'
## shape = tuple of training data shape
##
###################################################
def BuildModel(layers, shape):
    from keras.models import Model
    from keras.layers import Input, Dense, SimpleRNN, LSTM, Conv1D, Flatten, Dropout
    
    ins = Input(shape=shape[1:])
    hh = ins
    for ll, layer in enumerate(layers):
        if layer[0] is 'dnn':
            hh = Dense(layer[1], activation='tanh')(hh)
        elif layer[0] is 'rnn':
            hh = SimpleRNN(layer[1], activation='tanh', return_sequences=(ll+1<len(layers)) and (layers[ll+1][0] in ['cnn','lstm','rnn']))(hh)
        elif layer[0] is 'lstm':
            hh = LSTM(layer[1], activation='tanh', return_sequences=(ll+1<len(layers)) and (layers[ll+1][0] in ['cnn','lstm','rnn']))(hh)
        elif layer[0] is 'cnn':
            hh = Conv1D(layer[1], 3, padding='valid', activation='relu')(hh)
            if (ll+1 >= len(layers)) or (layers[ll+1][0] in ['dnn']):
                hh = Flatten()(hh)
        elif layer[0] is 'drop':
            hh = Dropout(layer[1])(hh)
    
    trade = Dense(5, activation='softmax')(hh)
    limit = Dense(1, activation='tanh')(hh)

    model = Model(inputs=ins, outputs=[trade, limit])
    model.compile(loss=['categorical_crossentropy','mse'], optimizer='adam')
    return model




###################################################
## Train model against provided data
##
###################################################
def TrainModel(model, sdf, dx, days=5, maxiter=1000):
    import matplotlib.pyplot as plt
    from keras.models import clone_model
    from stocksml import EvaluateChoices
    import time
    
    models = [model, clone_model(model)]
    models[1].compile(loss=['categorical_crossentropy','mse'], optimizer='adam')

    fig, ax = plt.subplots(2,2)
    last_plot = 0.0
    cum_results = np.zeros((maxiter,3))

    for ee in range(maxiter):
        
        # randomly select a week of data
        ss = np.random.randint(10,dx.shape[0]-6)
        dates = list(sdf.index.values[ss:ss + days])
        
        # each model makes a set of trades for the week
        results = np.zeros((len(models)))
        choices = [[] for _ in range(len(models))]
        for mm in range(len(models)):
            preds = models[mm].predict_on_batch(dx[ss:ss+days])
            choices[mm] = [(np.argmax(preds[0][dd]), preds[1][dd][0]) for dd in range(days)]
            
            # evaluate the performance of the trade choices
            results[mm] = EvaluateChoices(sdf, 'SPY', dates, choices[mm])

        # plot training every second
        if time.time() - last_plot > 1.0:
            for mm in range(2): ax[mm,0].clear()
            for mm in range(len(models)):
                rc = ax[0,0].scatter(np.arange(days), [cc[0] for cc in choices[mm]])
                rc = ax[1,0].scatter(np.arange(days), [cc[1] for cc in choices[mm]])
            plt.pause(0.05)
            last_plot = time.time()
        
        # the model earning the most money wins
        # the winner defines the truth data for this week
        # if neither is successful, skip training this week
        winner = np.argmax(results)
        cum_results[ee][0] = results[winner]
        if np.max(results) <= 1.0: continue
        if np.max(np.abs(np.diff(results))) < 0.0025: continue

        truth = [np.zeros((days,5), dtype=int), np.array([ll[1] for ll in choices[winner]]).reshape(-1,1).clip(-0.05,0.05)]
        for dd in range(days):
            truth[0][dd,choices[winner][dd][0]] = 1

        # train losing model with winners truth data if winner made money
        for mm in range(len(models)):
            if mm == winner: continue
            cum_results[ee][1:] = models[mm].train_on_batch(dx[ss:ss+5], truth)[1:]
            print('updated model %s'%str(mm), cum_results[ee][1:], results)
            
        # update the plots
        for mm in range(2): ax[mm, 1].clear()
        train_points = np.where(cum_results[:, 1] > 0)[0]
        rc = ax[0, 1].plot(np.arange(ee), cum_results[:ee, 0])
        rc = ax[1, 1].plot(train_points, cum_results[train_points, 1:])
        rc = ax[1, 1].set_yscale('log')
        plt.pause(0.05)


#########################################################
##
##
##
#########################################################
def Demo():
    import time
    from stocksml import FetchData, BuildData, Vectorize

    # Globals
    # SYMBOL = 'ARNC'
    SYMBOLS = ['SPY', 'BND']

    # TRAIN_START = '1992-01-01'
    TRAIN_START = '2017-01-01'
    TRAIN_END = time.strftime("%Y-%m-%d")

    # download data if necessary
    sdf = FetchData(SYMBOLS, TRAIN_START, TRAIN_END)

    # load data and build features
    ddf = BuildData(SYMBOLS)

    # format for model input
    dx = Vectorize(ddf.values, depth=5)

    model = BuildModel([('rnn',32),('dnn',64),('dnn',32)], dx.shape)

    TrainModel(model, sdf, dx, 5, 1000)
