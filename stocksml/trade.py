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


######################
def logger(params):
    logstr = '\nnull\n'
    if len(params) == 9:
        logstr = '  ->  %6s %4i shares at %6.1f  ($%2.1f, $%2.1f, %6.1f, %6.1f, %6.1f, %6.1f )' % params
    elif len(params) == 6:
        logstr = '\n%10s %5s %6s order for %4i shares of %4s at %6.1f' % params
    return logstr


###################################################
def EvaluateChoices(sdf, symbols, dates, choices, baseline=None):
    """
    Evaluate trading strategy choices
    
    Parameters
    ----------
    sdf : pandas.DataFrame
        symbol dataframe with price information
    symbols : list of str
        list of symbol tickers corresponding to the symbol enum in choices
    dates : list of str
        dates corresponding to choices, should match subset of pdf index values
    choices : list of tuples
        tuple of (action, symbol enum, limit) for each day. action is an enum of range 0-4 where
        [buy_limit, buy_sell, hold, sell_limit, sell_buy]. limit is the percent over/under open price (range -1 to 1)
    baseline : str
        ticker symbol to use for baseline buy-hold strategy. Default None will not compute a baseline (returns 0)
        
    Returns
    -------
    float, float, str
        performance of choices and baseline as a fraction of initial cash and ledger log of trades
    """
    cash = 1000
    shares = 0
    share_symbol, share_prices = '', 0
    log = ''
    reference = 0
    for ii in range(len(choices)):
        didx = sdf.index.get_loc(dates[ii])  # index of this date
        pds = sdf.iloc[didx+1]  # execute decision the next day
        date = sdf.index.values[didx+1]
        symbol = symbols[choices[ii][1]].lower()

        # establish baseline of buy and hold for the duration
        if (ii == 0) and (baseline is not None):
            reference = 1000 / pds[baseline.lower()+'_open']
        if (ii == len(choices)-1) and (baseline is not None):
            reference = (reference * pds[baseline.lower()+'_open'])/1000

        limit = pds[symbol+'_open'] * (1 + choices[ii][2])

        # store price data for the day (used by logger)
        prices = (pds[symbol + '_high'], pds[symbol + '_low'], pds[symbol + '_open'], pds[symbol + '_close'])
        if len(share_symbol) > 0:
            share_prices = (pds[share_symbol+'_high'], pds[share_symbol+'_low'], pds[share_symbol+'_open'], pds[share_symbol+'_close'])
            
        # buy limit - sell whatever is currently held at open price and buy new shares at limit
        if choices[ii][0] == 0:
            if shares > 0:
                log += logger((date, 'sell', 'market', shares, share_symbol, 0.0))
                market = pds[share_symbol+'_open']
                cash += shares * market
                log += logger(('sold', shares, market, cash, cash)+share_prices)
                shares = 0
                
            desired_shares = cash // limit
            log += logger((date, 'buy', 'limit', desired_shares, symbol, limit))
            if pds[symbol+'_low'] <= limit:
                market = min(limit, pds[symbol+'_open'])
                shares = desired_shares
                cash = cash - shares*market
                share_symbol = symbol
                log += logger(('bought', shares, market, cash, cash+shares*market)+prices)
            
        # buy open, sell limit - sell whatever is currently held at open, buy new shares at open and sell at limit
        elif choices[ii][0] == 1:
            if shares > 0:
                log += logger((date, 'sell', 'market', shares, share_symbol, 0.0))
                market = pds[share_symbol+'_open']
                cash += shares * market
                log += logger(('sold', shares, market, cash, cash)+share_prices)
            market = pds[symbol+'_open']
            shares = cash // market
            cash = cash - shares * market
            share_symbol = symbol
            log += logger((date, 'buy', 'market', shares, symbol, 0.0))
            log += logger(('bought', shares, market, cash, cash+shares*market)+prices)
            log += logger((date, 'sell', 'limit', shares, symbol, limit))
            if pds[symbol+'_high'] >= limit:
                market = max(limit, pds[symbol+'_open'])
                cash += shares * market
                log += logger(('sold', shares, market, cash, cash) + prices)
                shares = 0

        # sell limit - sell whatever is held at limit
        elif (choices[ii][0] == 3) and (shares > 0):
            limit = pds[share_symbol+'_open'] * (1 + choices[ii][2])
            log += logger((date, 'sell', 'limit', shares, share_symbol, limit))
            if pds[share_symbol+'_high'] >= limit:
                market = max(limit, pds[share_symbol+'_open'])
                cash += shares * market
                log += logger(('sold', shares, market, cash, cash) + share_prices)
                shares = 0

        # sell limit, buy close - sell whatever is currently held at the limit and buy new shares at close
        elif choices[ii][0] == 4:
            if shares > 0:
                limit = pds[share_symbol + '_open'] * (1 + choices[ii][2])
                log += logger((date, 'sell', 'limit', shares, share_symbol, limit))
                if pds[share_symbol+'_high'] >= limit:
                    market = max(limit, pds[share_symbol+'_open'])
                    cash += shares * market
                    log += logger(('sold', shares, market, cash, cash) + share_prices)
                    shares = 0
            if shares == 0:  # only buy if we are currently holding no other shares
                market = pds[symbol+'_close']
                shares = cash // market
                cash = cash - shares*market
                share_symbol = symbol
                log += logger((date, 'buy', 'market', shares, symbol, 0.0))
                log += logger(('bought', shares, market, cash, cash+shares*market) + prices)

    # liquidate at end
    if shares > 0:
        log += '\n---------- liquidate ----------'
        log += logger((date, 'sell', 'market', shares, share_symbol, 0.0))
        cash += shares * pds[share_symbol+'_close']
        log += logger(('sold', shares, pds[share_symbol+'_close'], cash, cash) +
                      (pds[share_symbol+'_high'], pds[share_symbol+'_low'], pds[share_symbol+'_open'], pds[share_symbol+'_close']))
    log += '\n---------- result = $%2.1f at %2.3f of baseline ----------\n' % (cash, (cash/1000)/reference if reference > 0 else 0)
    cash = cash / 1000
    
    return cash, reference, log
