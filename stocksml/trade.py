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



###################################################
## Evaluate trading strategy choices
##
## sdf = symbol data for range that choices were made
## symbol = ticker to use for trading
## dates = dates corresponding to choices
## choices = tuple of trading decision and limit for each day
##   decision = enum 0-4 = [buy_limit, buy_sell, hold, sell_limit, sell_buy]
##   limit = -1.0 to 1.0 = fraction of open for each choice
###################################################
def EvaluateChoices(pdf, symbol, dates, choices):
    
    # perform simple trading strategy against choices
    cash = 1000
    shares = 0
    for ii in range(len(choices)):
        didx = pdf.index.get_loc(dates[1])  # index of this date
        pds = pdf.iloc[didx+1]  # execute decision the next day

        limit = pds[symbol.lower() + '_open'] * (1 + choices[ii][1])

        # buy limit
        if (choices[ii][0] == 0) and (cash > 0) and (pds[symbol.lower() + '_low'] <= limit):
            shares = cash / limit
            cash = 0

        # buy open, sell limit
        elif (choices[ii][0] == 1) and (cash > 0):
            shares = cash / pds[symbol.lower() + '_open']
            cash = 0
            if pds[symbol.lower() + '_high'] >= limit:
                cash = shares * limit
                shares = 0

        # sell limit
        elif (choices[ii][0] == 3) and (shares > 0) and (pds[symbol.lower() + '_high'] >= limit):
            cash = shares * limit
            shares = 0

        # sell open, buy limit
        elif (choices[ii][0] == 4) and (shares > 0):
            cash = shares * pds[symbol.lower() + '_open']
            shares = 0
            if pds[symbol.lower() + '_low'] <= limit:
                shares = cash / limit
                cash = 0

    result = (cash + shares*pds[symbol.lower()+'_close'])/1000
    
    return result
