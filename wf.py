import pandas as pd
import numpy as np
import datetime


class Portfolio():

    def __init__(self, returns, params):

        self.returns = returns  # realized portfolio returns
        self.params = params
        self.cum_return = 0
        self._P0 = 1
        # the current vector of prices
        self.P = np.dot(np.ones([1, 5]), self._P0)

    def _roll_prod(self, x):
        return np.prod([1 + y for y in x])

    def buy_and_hold(self, A0, w0, r):
        '''
        Inputs:
            A0 - initial investment before trading day 0
            w0 - initial portfolio before trading day 0, 1-by-N, sums to 1
            r  - the matrix of realized returns, T-by-N pandas df

        Outputs:
            A1 - final period portfolio value
            w1 - final value-weighted portfolio weights
        '''
        if A0 > 0:
            x0 = np.dot(A0, w0)  # vector of market values of each security
            # return a vector of cumulative hodling returns
            R = [self._roll_prod(r[col].values) for col in r.columns]
            self.A, self.x = np.dot(x0, R), np.array(x0) * np.array(R)
            self.w = (self.x / self.A)
            self.P = self.P * R
        else:
            self.A, self.w = A0, w0

    def _deposit(self, D, alph):
        '''
        Inputs:
            D - the total amount of cash invested
            alph - weights for the new investment

        '''

        self.x = np.dot(self.A, self.w) + np.dot(D, alph)
        self.w = np.divide(self.x, D + self.A)
        self.A = self.A + D

    # def _rebalance(self, alph):
        # new position minus the current position, if negative, it is a sale
        #net_x = np.dot(self.A, alph) - self.x
        #sale_ind = net_x < 0

        # current position minus initial position of last purchase
        # if positive, a capital gain, o.w. capital loss
        #gain_ind = (self.x - self.x0) > 0
        #loss_ind = (self.x - self.x0) < 0

        #self.x = np.dot(self.A, alph)
        #self.w = alph
        # self.cum_return = self.cum_return +

    def trade(self, alph, types, D=0):
        '''
        Trade function takes in the target portfolio weights, states of the
        portfolio, and returns the updated portfolio characteristics
        '''

        if types == "deposit":
            self._deposit(self, D, alph)

    def weight_path(self, A0, w0, r):
        '''
        Compute the path of the portfolio weights:
        Inputs:
            A0 - initial investment before trading day 0, scalar
            w0 - initial portfolio before trading day 0, 1-by-N list, sums to 1
            r  - the matrix of realized returns from the closing of day 0 to the final date of r

        Outputs:
            path - the full path of realized wegiths from day 0 to the final date of r 

        '''

        path_w = pd.DataFrame(index=r.index, columns=r.columns)
        oneday = datetime.timedelta(days=1)
        self.A = A0
        self.w = w0
        P0 = self.P

        for date in r.index:
            self.buy_and_hold(self.A, self.w, r.loc[date:date + oneday, :])
            path_w.loc[date, :] = self.w
        self.A = A0  # restore values
        self.w = w0
        self.P = P0

        return path_w

    def current_portfolio(self, A0, w0, r, path=False):
        oneday = datetime.timedelta(days=1)
        initial_date = r.index[0]
        self.A = A0
        self.w = w0
        # the current vector of prices
        self.P = np.dot(np.ones([1, 5]), self._P0)
        self.p = pd.DataFrame(index=r.index, columns=r.columns)
        self.s = pd.DataFrame(index=r.index, columns=r.columns)
        self.tau = pd.DataFrame(index=r.index, columns=r.columns)

        if path:
            w_path = pd.DataFrame(index=r.index, columns=r.columns)

        for key, val in self.inv_amt.items():
            
            _inv_date = datetime.datetime.strptime(key, '%Y-%m-%d')
            _r = r.loc[initial_date: _inv_date + oneday, :]
            
            # get the simulated path of the portfolio till the next investment
            if path:
                w_path.loc[initial_date: _inv_date +
                           oneday, :] = self.weight_path(self.A, self.w, _r)
                
            # get the buy-and-hold result before the current investment
            self.buy_and_hold(self.A, self.w, _r) # change the price, the market value, and the weight to next investment date
            self.p.loc[_inv_date, :] = self.P
            self.s.loc[_inv_date, :] = np.floor(val / self.P)
            
            # determine whether this is a long-term or short-term invesment
            if (r.index[-1] - _inv_date) > datetime.timedelta(days = 260):
                self.tau.loc[_inv_date, :] = np.dot(np.ones([1, 5]), 0.2)
            else:
                self.tau.loc[_inv_date, :] = np.dot(np.ones([1, 5]), 0.3)

            # update the portfolio after more deposits
            self._deposit(val, self.inv_alph[key])
            initial_date = _inv_date + oneday

        self.buy_and_hold(self.A, self.w, r.loc[initial_date:, :])
        self.p.loc[r.index[-1], :] = self.P
        self.p = self.p.dropna()
        self.s = self.s.dropna()
        self.tau = self.tau.dropna()

        if path:
            w_path.loc[initial_date:, :] = self.weight_path(
                self.A, self.w, r.loc[initial_date:, :])

        if path:
            return w_path

class Rebalancing():
    
    def __init__(self, gamma, wbar, Pf):
        self.gamma = gamma
        self.p = Pf.p.loc[Pf.returns.index[-1], :].values
        self.s = Pf.s
        self.c = Pf.p.loc[Pf.p.index[:-1], :]
        self.tau = Pf.tau   
        self.wbar = wbar
    
    # security-level optimal strategy
    def single_tax(self, delta_x, p, state_s, state_p, state_tau):
        '''
        The optimal trading strategy of a single security
        Inputs: 
            - delta_x: the amount of investment/liquidation of a single security
            - p: current price of the security
            - state_s: the existing stock positions
            - state_p: the prices of the existing position
            - state_tau: the tax rates of the existing position
        '''

        if delta_x < 0:
            delta_p = p - state_p
            mctacg = delta_p * state_tau # marginal tax-adjusted cpaital gains 
            ordered = np.argsort(mctacg) # get the indices from the largest marginal loss to the smallest margianl loss
            delta_s = np.zeros(len(state_s))
            used_s = 0

            for i in ordered:
                num_sell = min(state_s[i], abs(delta_x)/p - used_s)
                delta_s[i] = num_sell
                used_s = num_sell + used_s

            tax = np.sum(delta_s * delta_p * state_tau)

        else:
            tax = 0

        return tax

    # the total tax under the security-level optimal strategy and an allocation scheme x
    def T(self, x):
        '''
        Extend the security-level strategy to the portfolio level:
        Inputs:
            - x: the vector of delta_x
            - p: current prices 
            - s: existing postions
            - c: costs
            - tau: tax rates
        '''
        taxes = []
        p = self.p
        s = self.s
        c = self.c
        tau = self.tau

        for i in range(len(c.columns)):
            state_p = c[c.columns[i]].values
            state_s = s[c.columns[i]].values
            state_tau = tau[c.columns[i]].values

            taxes.append(self.single_tax(x[i], p[i], state_s, state_p, state_tau))

        totaltax = np.array(taxes).sum()
        return totaltax 

    def alph(self, x):

        market_val = []
        p = self.p
        s = self.s
        c = self.c
        tau = self.tau
        
        for i in range(len(c.columns)):
            state_s = s[c.columns[i]].values
            total_s = state_s.sum()
            old_val = total_s * p[i] # old market value 
            new_val = old_val + x[i] # new market value after rebalancing

            market_val.append(new_val)

        w = np.array(market_val) / np.array(market_val).sum()
        return w, market_val

    def V(self, x):

        '''
        The objective function, as function of the rebalancing allocation vector, 
        '''
        alph, xbar = self.alph(x)

        # chooes a proper gamma so that the tracking error is important enough
        y = self.T(x) + self.gamma * np.sum(np.square(alph - self.wbar))
        return y

