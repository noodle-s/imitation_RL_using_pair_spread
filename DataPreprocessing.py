import warnings
import pandas as pd
import pandas_datareader.data as web
import os
from arch.unitroot import engle_granger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import deque

stock_type = {
    'kospi': 'stockMkt',
    'kosdaq': 'kosdaqMkt'
}


# get ticker
class Ticker:
    def __init__(self):
        super(Ticker, self).__init__()
        self.ticker = None
        self._download_ticker()

    # get data url
    def _get_download_url(self, market_type=None):
        market_type = stock_type[market_type]
        download_link = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download'
        download_link = download_link + '&marketType=' + market_type
        download_link = pd.read_html(download_link, header=0)[0]

        return download_link

    def _download_ticker(self):
        def _get_download_kospi():
            kospi_ticker = self._get_download_url('kospi')

            return kospi_ticker

        def _get_download_kosdaq():
            kosdaq_ticker = self._get_download_url('kosdaq')

            return kosdaq_ticker

        kospi_ticker = _get_download_kospi()
        kosdaq_ticker = _get_download_kosdaq()

        self.ticker = pd.concat([kospi_ticker, kosdaq_ticker])
        self.ticker = self.ticker.rename(columns={'종목코드': 'ticker'})
        self.ticker = self.ticker['ticker'].map('{:06d}'.format).tolist()


class DataGet(Ticker):
    def __init__(self, end_date=None, params=None):
        super().__init__()
        self.start_date = params['period_start']
        self.end_date = end_date
        self.dist = params['dist']
        self.daily_data = pd.DataFrame()
        self.kospi_200 = pd.DataFrame()
        self.kospi_200_list = pd.DataFrame()
        self.target_stock = params['target']
        self.mdpp_rate = params['mdpp_rate']
        self.turning_point = []
        self.pair = pd.DataFrame()
        self.train_data = {}
        self.open_target = pd.Series()
        self.form_period_end = params['form_period_end']
        self.demo_period_end = params['demo_period_end']
        self.test_period_end = params['test_period_end']
        self.test_data = {}
        self.n_step = params['n_step']
        self.tax = params['tax']
        self.transaction_cost = params['transaction_cost']
        self.discount_factor = params['discount_factor']
        self.order_rate = params['order_rate']
        self.nrm = params['nrm']
        self.init_bal = params['bal']
        self.expert_action = params['expert_action']

    def get_daily_data(self):
        if os.path.isfile("./daily_data.csv"):
            self.daily_data = pd.concat([pd.read_csv("./daily_data_2.csv"),
                                         pd.read_csv("./daily_data.csv")])

        else:
            for ticker in self.ticker:
                tmp_daily_data = web.DataReader(str(ticker), 'naver', self.start_date, self.end_date)
                tmp_daily_data['ticker'] = str(ticker)
                self.daily_data = pd.concat([self.daily_data, tmp_daily_data], axis=0)
            self.daily_data = self.daily_data.to_csv("./daily_data.csv")

        self.daily_data = self.daily_data.sort_values(by=['ticker', 'Date'])

    def get_min_data(self):
        pass

    # get all kospi 200 data
    def get_kospi_200_data(self):
        self.get_daily_data()

        if os.path.isfile("./KOSPI_200.xlsx"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self.kospi_200_list = pd.read_excel("./KOSPI_200.xlsx", engine="openpyxl")
            self.kospi_200_list = self.kospi_200_list.rename(columns={'종목코드': 'ticker', '종목명': 'name'})
            self.kospi_200_list = self.kospi_200_list[['ticker', 'name']]
            self.kospi_200_list['ticker'] = self.kospi_200_list['ticker'].map('{:06d}'.format)

        self.daily_data['ticker'] = self.daily_data['ticker'].map('{:06d}'.format)
        self.kospi_200 = self.daily_data[self.daily_data['ticker'].isin(self.kospi_200_list['ticker'])]
        self.kospi_200 = pd.merge(self.kospi_200, self.kospi_200_list, on='ticker')
        self.kospi_200 = self.kospi_200[self.start_date <= self.kospi_200['Date']]
        self.kospi_200 = self.kospi_200.set_index('Date')

    def _get_moving_average(self, train=False):
        if train:
            open_target = copy.deepcopy(self.open_target[self.open_target.index <= self.form_period_end])

        else:
            open_target = copy.deepcopy(self.open_target[self.open_target.index > self.form_period_end])

        target = pd.DataFrame()
        target[f'{self.target_stock}_5MA'] = open_target.rolling(window=5, min_periods=1).mean()
        target[f'{self.target_stock}_20MA'] = open_target.rolling(window=20, min_periods=1).mean()
        target[f'{self.target_stock}_60MA'] = open_target.rolling(window=60, min_periods=1).mean()
        target[f'{self.target_stock}_120MA'] = open_target.rolling(window=120, min_periods=1).mean()
        target[f'{self.target_stock}_240MA'] = open_target.rolling(window=240, min_periods=1).mean()

        return target

    def preprocessing_data(self):
        self.get_kospi_200_data()

        open_target = self.kospi_200[(self.kospi_200['name'] == self.target_stock)]['Open']
        self.open_target = open_target[(self.start_date <= open_target.index) & (open_target.index <= self.test_period_end)]
        self.open_target = self.open_target.mask(self.open_target == 0).ffill(downcast='infer')

        target_ma = self._get_moving_average(train=True)
        state, pair_dict = self._engle_granger_test()
        state = pd.concat([state, target_ma], axis=1)
        state, scaler = self._scaling(state)
        state['holdings'] = 0
        state['number_of_holdings'] = 0
        state['rate'], test_rate, test_turning_point = self._get_tur()
        state['turning_point'] = self.turning_point

        data = {'state': state,
                'n_step_state': np.zeros((state.shape[0] - (self.n_step - 1), self.n_step, state.shape[1]))}

        self.train_data, _, _ , _= self._get_transition(state, data, demo=False)
        self._get_test_data(pair_dict, scaler, test_rate, test_turning_point)
        self._draw_state_plot(pair_dict, state)
        self.pair = pair_dict

    def _scaling(self, state):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(state)
        data = pd.DataFrame(data, columns=state.columns, index=state.index)

        return data, scaler

    def _get_test_data(self, pair_dict, scaler=None, test_rate=None, test_turning_point=None):
        del pair_dict[self.target_stock]

        pair_list = list(pair_dict.keys())
        value = self.kospi_200[(self.kospi_200.index <= self.test_period_end)
                               & (self.kospi_200.index > self.form_period_end)][['Open', 'name']]
        target = self.open_target[self.open_target.index > self.form_period_end]
        state = pd.DataFrame()

        for idx, pair in enumerate(pair_list):
            pair_value = value[value['name'] == pair]['Open']
            pair_value = pair_value.mask(pair_value.values == 0).ffill(downcast='infer')
            spread = pd.Series(np.log(pair_value / target), name=pair + " spread")
            price = pd.Series(pair_value, name=pair + " price")
            state = state.append([spread, price])

        state = state.transpose()
        state[self.target_stock] = target
        target_ma = self._get_moving_average()
        state = pd.concat([state,target_ma], axis=1)
        state = pd.DataFrame(scaler.transform(state), columns=state.columns, index=state.index)
        state['holdings'] = 0
        state['number_of_holdings'] = 0
        state['rate'] = test_rate
        state['turning_point'] = test_turning_point

        data = {'state': state,
                'n_step_state': np.zeros((state.shape[0] - (self.n_step - 1), self.n_step, state.shape[1]))}

        self.test_data, _, _, _ = self._get_transition(state, data, demo=False)

    def get_state(self):
        train_state = self.train_data['state'].iloc[self.n_step:, :]
        test_state = self.test_data['state']
        train_n_step_state = self.train_data['n_step_state'][self.n_step:, :, :]

        tmp = self.get_tail_data(train_state[-self.n_step + 1:], test_state[:self.n_step - 1], self.n_step)
        test_n_step_state = np.concatenate((tmp, self.test_data['n_step_state']), axis=0)
        train_state = train_state.iloc[self.n_step - 1:, :]
        train_state = train_state[train_state.index > self.demo_period_end]
        train_n_step_state = train_n_step_state[-len(train_state):, :, :]

        return train_state, train_n_step_state, test_state, test_n_step_state

    def get_tail_data(self, train_tail, test_head, n_step):
        concat = pd.concat([train_tail, test_head])
        tmp = np.zeros((n_step - 1, n_step, test_head.shape[1]))
        for i in range(n_step - 1, len(concat)):
            tmp[i - (n_step - 1), :, :] = concat.iloc[(i - (n_step - 1)): i + 1]

        return tmp

    # engle granger cointegration test
    def _engle_granger_test(self):
        target = self.open_target[self.open_target.index <= self.form_period_end]
        pair_dict = {self.target_stock: target}
        state = pd.DataFrame()

        for stock_name in self.kospi_200_list['name']:
            value = self.kospi_200[self.kospi_200.index <= self.form_period_end]
            value = value[value['name'] == stock_name]['Open']
            value = value.mask(value == 0).ffill(downcast='infer')

            try:
                test = engle_granger(target, value)
                if test.pvalue <= 0.05 and 0 not in value:
                    pair_dict[stock_name] = value
                    spread = pd.Series(np.log(value / target), name=stock_name + " spread")
                    price = pd.Series(value, name=stock_name + " price")
                    state = state.append([spread, price])

            except:
                continue

        state = state.transpose()
        state[self.target_stock] = target
        return state, pair_dict

    # minimum distance/percentage principle
    def _get_tur(self):
        data = copy.deepcopy(self.open_target)
        test_data = data[data.index > self.form_period_end]
        turning_point = np.zeros(len(data))
        rate = np.zeros(len(data))

        if self.expert_action == 'MDPP':
            turning_point, rate = self._mdpp(data, rate, turning_point)

        elif self.expert_action == 'Osciliator':
            turning_point, rate = self._osciliator(data)

        elif self.expert_action == 'RSI':
            turning_point, rate = self._rsi(data, rate, turning_point)

        else:
            turning_point, rate = self._macd(data)

        test_turning_point = turning_point[-len(test_data):]
        self.turning_point = turning_point[:-len(test_data)]

        return rate[:-len(test_data)], rate[-len(test_data):], test_turning_point

    def _mdpp(self, data, rate, tur):
        for i in range(self.dist, len(data)):
            before = data[i - self.dist]
            now = data[i]
            rate_tmp = abs(now - before) / ((abs(before) + abs(now)) / 2)
            rate[i] = rate_tmp * 100

            if rate[i] > self.mdpp_rate:
                if before > now:
                    tur[i - self.dist] = 1  # sell
                    tur[i] = 2  # buy

                else:
                    tur[i - self.dist] = 2  # buy
                    tur[i] = 1  # sell

        return tur, rate

    def _osciliator(self, data):
        stochastic = np.zeros(len(data))

        for i in range(15, len(data)):
            low_price = min(data.values[i-15:i])
            now = data.values[i]
            high_price = max(data.values[i-15:i])

            if high_price - low_price == 0:
                rate_tmp = 0

            else:
                rate_tmp = (now - low_price) / (high_price - low_price)

            stochastic[i] = rate_tmp * 100

        stochastic = pd.Series(stochastic)
        k_line = stochastic.rolling(window=5, min_periods=1).mean()
        d_line = k_line.rolling(window=3, min_periods=1).mean()
        rate = k_line - d_line

        tur = rate.map(lambda e: 2 if e > 0 else (1 if e < 0 else 0))
        tur = tur.mask(tur.diff().fillna(0) == 0, 0)

        return tur, rate

    def _rsi(self, data, rate, tur):
        price = copy.deepcopy(data).diff().fillna(0)

        for i in range(15, len(data)):
            upward = sum(price[i-15:i].map(lambda e: e > 0)) / 15
            downward = sum(price[i-15:i].map(lambda e: e < 0)) / 15

            if downward == 0:
                downward = 1

            rs = upward / downward
            rate[i] = 100 * rs / (1 + rs)

            if rate[i] > 70: # 70, 50
                tur[i] = 1  # sell

            elif rate[i] < 30: # 50, 30
                tur[i] = 2 # buy

            else:
                tur[i] = 0 # none

        tur = pd.Series(tur)
        tur = tur.mask(tur.diff().fillna(0) == 0, 0)

        return tur, rate

    def _macd(self, data):
        price = copy.deepcopy(data)
        ma_12 = price.rolling(window=12, min_periods=1).mean()
        ma_26 = price.rolling(window=26, min_periods=1).mean()
        macd = ma_12 - ma_26
        signal = macd.rolling(window=9, min_periods=1).mean()
        osciliator = macd - signal

        tur = osciliator.map(lambda e: 2 if e > 0 else (1 if e < 0 else 0))
        tur = tur.mask(tur.diff().fillna(0) == 0, 0)

        return tur, osciliator

    def _get_reward(self, info, action):
        now = info["trading_price"][-1]
        before = info["trading_price"].popleft()
        reward = action * np.log(now / before) * 10

        return reward

    def _get_balance_price(self, action, current_price, number_of_orders):
        if action == -1:
            price = current_price * number_of_orders + \
                    (current_price * number_of_orders * self.transaction_cost)

        else:
            price = current_price * number_of_orders \
                    - ((current_price * number_of_orders * self.transaction_cost)
                       + (current_price * number_of_orders * self.tax))

        return price

    def _buy_action(self, current_price, before_price, info, action, number_of_orders, start=False):
        buy_price = self._get_balance_price(action, current_price, number_of_orders)
        info["trading_price"].append(current_price)
        info["bal"] -= buy_price

        if info["holdings"] == 0:
            reward = 0
            # if start:
            #     reward = 0
            #
            # else:
            #     reward = action * np.log(current_price / before_price) * 100

        else:
            if info["trading_price"][-1] == info["trading_price"][-2]:
                reward = 0

            else:
                reward = self._get_reward(info, action)

        info['holdings'] += number_of_orders

        return info, reward

    def _sell_action(self, current_price, before_price, info, action, number_of_orders):
        if info["holdings"] != 0:  # stock holding
            info["trading_price"].append(current_price)

            if info["holdings"] < number_of_orders:
                number_of_orders = info["holdings"]

            sell_price = self._get_balance_price(action, current_price, number_of_orders)
            info["bal"] += sell_price

            if info["trading_price"][-1] == info["trading_price"][-2]:
                reward = 0

            else:
                reward = self._get_reward(info, action)

            info["holdings"] -= number_of_orders

            if info["holdings"] <= 0:
                info["trading_price"] = deque(maxlen=2)

        else:
            reward = -0.01
            # reward = action * np.log(current_price / before_price) * 10

        return info, reward

    def _none_action(self, rate, info):
        if rate >= self.mdpp_rate:
            reward = 0

        else:
            reward = 0

        return info, reward

    def _get_demo_transition(self, info, data, state, target, action_list):
        def _set_demo_env(index, data, info, target, state, action):
            number_of_orders = int(np.trunc((self.init_bal * self.order_rate) / target[index]))

            if index < state.shape[0] - 1:
                if info["holdings"]:
                    data['state'].iloc[index, -3] = 1  # holdings
                    data['state'].iloc[index, -2] = info['holdings']  # number of holdings

                if action == -1:  # buy
                    info, reward = self._buy_action(target[index], target[index-1], info, action, number_of_orders)

                elif action == 1:  # sell
                    info, reward = self._sell_action(target[index], target[index-1], info, action, number_of_orders)

                else:
                    info, reward = self._none_action(state['rate'][index], info)

                if reward < 0:
                    reward = reward * self.nrm

                info["holding_price"] = target[index] * info['holdings']
                data['reward'][index] = self.discount_factor * reward
                data['next_state'][index, :] = data['state'].iloc[index + 1, :]
                data['terminate'][index] = 0

                if info["holdings"]:
                    data['next_state'][index, -3] = 1  # holdings
                    data['next_state'][index, -2] = info['holdings']  # number of holdings

            else:
                if info["holdings"]:
                    data['state'].iloc[index, -3] = 1  # holdings
                    data['state'].iloc[index, -2] = info['holdings']  # number of holdings

                data['reward'][index] = 1
                data['next_state'][index, :] = 0
                data['terminate'][index] = 1

                if info["holdings"]:
                    data['next_state'][index, -3] = 1  # holdings
                    data['next_state'][index, -2] = info['holdings']  # number of holdings

            return data

        def _get_n_step_state(index, data):
            data['n_step_state'][index - (self.n_step - 1), :, :] \
                = data['state'].iloc[index - (self.n_step - 1): index + 1, :]
            data['n_step_next_state'][index - (self.n_step - 1), :, :] \
                = data['next_state'][index - (self.n_step - 1): index + 1, :]

            return data

        def _get_result_bal(info):
            if info["bal"] != self.init_bal:
                if info["holding_price"] != 0:
                    info["result"].append(info["bal"] + info["holding_price"])

                else:
                    info["result"].append(info["bal"])

            else:
                info["result"].append(info["bal"])

        def _get_result_profit(result):
            profit_rate = np.round(((result[-1] / self.init_bal) - 1) * 100, 4)

            return profit_rate

        def _get_profit_rate(info):
            result = pd.Series(copy.deepcopy(info["result"]))
            profit_rate = np.round(((result.diff().fillna(0) / result.shift().fillna(0)) * 100).fillna(0), 4)

            return profit_rate

        def _get_sharpe_ratio(profit_rate):
            annualization_factor = 252 * 9
            rl_sharpe_ratio = (profit_rate.expanding().mean() / profit_rate.expanding().std() * np.sqrt(
                annualization_factor)).fillna(0)

            return rl_sharpe_ratio

        def _get_VWR(prices, MAV=0.5, TAU=2):
            ### 0. Alias
            annualization_factor = 252 * 9
            T = len(prices.values)

            ### 1. Compound Annual Growth (CAG)
            mean_log_return = np.log(prices.values[-1] / prices.values[0]) / (T - 1)
            CAGR = np.exp(mean_log_return * annualization_factor) - 1
            norm_return = CAGR * 100

            ### 2. Ideal spreads_val
            ideal_preds_val = [prices.values[0] * np.exp(mean_log_return * idx_time) for idx_time in range(T)]

            ### 3. Difference between ideal and real
            diff = [prices.values[idx_time] / ideal_preds_val[idx_time] - 1 for idx_time in range(T)]

            ### 4. Variability
            variability = np.std(diff, ddof=1)

            if variability < MAV:
                if norm_return > 0:
                    return dict(VWR=norm_return * (1 - (variability / MAV) ** TAU), CAGR=CAGR,
                                variability=variability)
                else:
                    return dict(VWR=norm_return * (variability / MAV) ** TAU, CAGR=CAGR,
                                variability=variability)
            else:
                return dict(VWR=0, CAGR=CAGR, variability=variability)

        for index in range(state.shape[0]):
            action = action_list[index]

            if action == 2:
                action = -1

            data = _set_demo_env(index, data, info, target, state, action)

            if index != state.shape[0]:
                _get_result_bal(info)

            if index >= self.n_step - 1:
                data = _get_n_step_state(index, data)

        vwr = np.round(_get_VWR(pd.Series(info["result"]))['VWR'], 4)
        profit_rate = _get_profit_rate(info)
        result_profit = _get_result_profit(info["result"])
        sharpe_ratio = _get_sharpe_ratio(profit_rate)

        print(f"demo vwr = {vwr}")
        print(f"demo sharpe ratio = {sharpe_ratio.values[-1]}")
        print(f"demo profit rate = {result_profit}")
        print(f"demo rest number of stocks = {info['holdings']}")

        return data, result_profit, np.round(sharpe_ratio.values[-1], 4), vwr

    def _get_transition(self, state, data, action_list=None, target=None, nrm=1, demo=True):
        profit_rate = None
        sharpe_ratio = None
        vwr = None

        def _get_trading_info():
            info = {'holdings': 0, 'trading_price': deque(maxlen=2), 'bal': self.init_bal, 'holding_price': 0, 'result': [self.init_bal]}

            return info

        if demo:
            info = _get_trading_info()
            data, profit_rate, sharpe_ratio, vwr = self._get_demo_transition(info, data, state, target, action_list)

        else:
            for index in range(state.shape[0]):
                if index >= self.n_step - 1:
                    data['n_step_state'][index - (self.n_step - 1), :, :] \
                        = data['state'].iloc[index - (self.n_step - 1): index + 1, :]

        return data, profit_rate, sharpe_ratio, vwr

    # demonstration data split
    def get_demo_data(self):
        state = copy.deepcopy(self.train_data['state'])
        state = state[state.index <= self.demo_period_end]
        state = state.iloc[self.dist:]
        target = self.open_target[self.open_target.index <= self.demo_period_end]
        action_list = np.array(self.turning_point[:len(target)]).astype(np.int64)
        action_list = action_list[self.dist:]
        target = target[self.dist:]

        demo_data = {'state': state,
                     'action': action_list,
                     'reward': np.array(np.zeros(state.shape[0])),
                     'next_state': np.zeros((state.shape[0], state.shape[1])),
                     'terminate': np.zeros(state.shape[0]),
                     'n_step_state': np.zeros((state.shape[0] - (self.n_step - 1), self.n_step, state.shape[1])),
                     'n_step_next_state': np.zeros((state.shape[0] - (self.n_step - 1), self.n_step, state.shape[1]))
                     }

        demo_data, profit_rate, sharpe_ratio, vwr = self._get_transition(state, demo_data, action_list, target, demo=True)
        demo_data = self._data_arange(demo_data)
        self._draw_demo_plot(target, action_list, profit_rate, sharpe_ratio, vwr)
        self._draw_action_rate(action_list)
        return demo_data

    def _data_arange(self, data):
        data['state'] = data['state'].iloc[(self.n_step - 1):, :]
        data['action'] = data['action'][(self.n_step - 1):]
        data['reward'] = data['reward'][(self.n_step - 1):]
        data['next_state'] = data['next_state'][(self.n_step - 1):, :]
        data['terminate'] = data['terminate'][(self.n_step - 1):]

        return data

    def _get_n_step_reward(self, reward):
        n_step_reward = 0
        reward = copy.deepcopy(reward)

        for i in range(len(reward)):
            if reward[i] == 0:
                continue
            tmp = reward[i] * (self.discount_factor ** i)
            n_step_reward += tmp

        return n_step_reward

    # draw pair price
    def _draw_state_plot(self, pair, state):
        plt.rcParams["font.family"] = 'NanumGothic'
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.rc('axes', unicode_minus=False)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        def _price_plot(ax):
            for stock_name in list(pair.keys()):
                value = np.array(list(pair[stock_name])).reshape(-1)
                ax.plot(pd.to_datetime(state.index), value, label=stock_name)

            ax.plot(pd.to_datetime(state.index),
                     self.open_target[self.open_target.index <= self.form_period_end].values,
                     label=self.target_stock)

            ax.set_title(f'formation Price Comparison (target = {self.target_stock})')
            ax.set_xlabel('date')
            ax.set_ylabel('Open')
            ax.legend()

        def _spread_plot(ax):
            state_columns = pd.Series(state.columns.values)
            state_columns = state_columns[state_columns.str.contains('spread')]

            for stock_name in state_columns.values:
                ax.plot(pd.to_datetime(state.index), state[stock_name], label=stock_name)

            ax.set_title(f'formation Spread Comparison (target = {self.target_stock})')
            ax.set_xlabel(f'date')
            ax.set_ylabel(f'Open spread')
            ax.legend()

        fig, (ax1, ax2) = plt.subplots(2, 1)
        _price_plot(ax1)
        _spread_plot(ax2)

        fig.savefig(f'./image/pair_open_{self.target_stock}_formation.png')
        plt.tight_layout()
        plt.show()
        plt.close()

    def _draw_demo_plot(self, data, action_list, profit_rate, sharpe_ratio, vwr):
        x_date = pd.to_datetime(data.index)
        y_data = np.array(data)

        tur = copy.deepcopy(action_list)
        buy = y_data[np.where(tur == 2)]
        sell = y_data[np.where(tur == 1)]

        buy_date = pd.to_datetime(x_date[np.where(tur == 2)])
        sell_date = pd.to_datetime(x_date[np.where(tur == 1)])

        plt.rcParams["font.family"] = 'NanumGothic'
        plt.rcParams["figure.figsize"] = (15, 6)
        plt.rc('axes', unicode_minus=False)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.plot(x_date, y_data)
        plt.plot(buy_date, buy, '^', color='r', markersize=4)
        plt.plot(sell_date, sell, 'v', color='b', markersize=4)

        print(f"number of buy date = {len(buy_date)}")
        print(f"number of sell date = {len(sell_date)}")

        plt.title(f'{self.expert_action}_{self.target_stock}_profit = {profit_rate}_sharpe ratio = {sharpe_ratio}_vwr = {vwr}')
        plt.xlabel('date')
        plt.ylabel('Open')
        plt.savefig(f'./image/{self.expert_action}_exam_{self.target_stock}_profit = {profit_rate}_sharpe ratio = {sharpe_ratio}_vwr = {vwr}.png')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    def _draw_action_rate(self, action):
        tur = copy.deepcopy(action)
        x = np.arange(3)
        buy_rate = round(np.count_nonzero(tur == 2) / len(tur), 4) * 100
        sell_rate = round(np.count_nonzero(tur == 1) / len(tur), 4) * 100
        none_rate = round(np.count_nonzero(tur == 0) / len(tur), 4) * 100
        action_dict = {"buy": buy_rate, "sell": sell_rate, "none": none_rate}

        p1 = plt.bar(x, action_dict.values())
        plt.legend()
        plt.bar_label(p1, label_type='center')
        plt.xticks(x, action_dict.keys())
        plt.ylabel('action rate(%)')
        plt.title(f"{self.expert_action}_exam_{self.target_stock}_action_rate.png")
        plt.savefig(f'./image/{self.expert_action}_exam_{self.target_stock}_action_rate.png')
        plt.tight_layout()
        plt.show()
        plt.close()