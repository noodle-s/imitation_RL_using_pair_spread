import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import copy
from collections import deque
gym.logger.set_level(40)


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, state, n_step_state, open, params, time, train):
        super(gym.Env, self).__init__()

        self.reward = []
        self.dist = params['dist']
        self.open = open
        self.target = params['target']
        self.transaction_cost = params['transaction_cost']
        self.tax = params['tax']
        self.holdings = 0
        self.current_step = 0
        self.trading_price = deque(maxlen=2)
        self.holding_price = 0
        self.current_buy = 0
        self.n_step = params['n_step']
        self.state = state
        self.n_step_state = n_step_state
        self.train = train
        self.action_list = []
        self.init_bal = params['bal']
        self.bal = self.init_bal
        self.profit_rate = pd.Series()
        self.nrm = params['nrm']
        self.time = time
        self.mdpp_rate = params['mdpp_rate']
        self.result = [self.init_bal]
        self.n_holdings = deque(np.zeros(self.n_step), maxlen=self.n_step)
        # Actions of the format Buy or Sell or None.
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([2]), dtype=np.float32)

        # Observation contains the spread and rate value
        self.observation_space = spaces.Box(
            low=-100, high=100000, shape=(1, state.shape[-1]), dtype=np.float32)

        self.dir_path = params['dir_path']
        self.order_rate = params['order_rate']
        self.discount_factor = params['discount_factor']

        self._get_open()

    def _get_open(self):
        self.open = self.open.iloc[(self.open.index >= self.state.index[0])
                                   & (self.open.index <= self.state.index[-1])]

    def _next_observation(self):
        # Get the stock data points before distance
        state = copy.deepcopy(self.state.iloc[self.current_step])
        n_step_state = copy.deepcopy(np.array([self.n_step_state[self.current_step, :, :]]))

        if self.holdings:
            state['holdings'] = 1
            state['number_of_holdings'] = self.holdings

            self.n_holdings.popleft()
            self.n_holdings.append(self.holdings)

        n_step_state[:, :, -2] = list(map(lambda x: 0 if x == 0 else 1, self.n_holdings))
        n_step_state[:, :, -1] = self.n_holdings

        return np.array(state), n_step_state

    def _get_balance_price(self, action, current_price, number_of_orders):
        if action == -1:
            price = current_price * number_of_orders + \
                        (current_price * number_of_orders * self.transaction_cost)

        else:
            price = current_price * number_of_orders \
                         - ((current_price * number_of_orders * self.transaction_cost)
                            + (current_price * number_of_orders * self.tax))

        return price

    def _get_reward(self, action):
        now = self.trading_price[-1]
        before = self.trading_price.popleft()

        reward = action * np.log(now / before) * 10
        # reward = ((np.sqrt(self.bal[-1]) - np.sqrt(self.bal[-2])) \
        #          / (self.trading_price[-1] - self.trading_price[-2]))

        return self.discount_factor * reward

    def _buy_action(self, action, current_price, before_price, number_of_orders):
        buy_price = self._get_balance_price(action, current_price, number_of_orders)
        self.trading_price.append(current_price)
        self.bal -= buy_price

        if self.holdings == 0:
            reward = 0
            # if self.current_step == 0:
            #     reward = 0
            #
            # else:
            #     reward = action * np.log(current_price / before_price) * 10

        else:
            if self.trading_price[-1] == self.trading_price[-2]:
                reward = 0

            else:
                reward = self._get_reward(action)

        self.holdings += number_of_orders

        return reward

    def _sell_action(self, action, current_price, before_price, number_of_orders):
        if self.holdings != 0:  # stock holding
            self.trading_price.append(current_price)

            if self.holdings < number_of_orders:
                number_of_orders = self.holdings

            sell_price = self._get_balance_price(action, current_price, number_of_orders)
            self.bal += sell_price

            if self.trading_price[-1] == self.trading_price[-2]:
                reward = 0

            else:
                reward = self._get_reward(action)

            self.holdings -= number_of_orders

            if self.holdings <= 0:
                self.trading_price = deque(maxlen=2)
                self.holdings = 0

        else:
            reward = -0.01
            # if self.current_step == 0:
            #     reward = 0
            # else:
            #     reward = action * np.log(current_price / before_price) * 10

        return reward

    def _none_action(self):
        if self.state['rate'][self.current_step] >= self.mdpp_rate:
            reward = 0

        else:
            reward = 0

        return reward

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.open.iloc[self.current_step]

        if self.current_step != 0:
            before_price = self.open.iloc[self.current_step-1]

        else:
            before_price = None

        number_of_orders = int(np.trunc((self.init_bal * self.order_rate) / current_price))

        if action == 2:  # Buy stock(1)
            action = -1
            reward = self._buy_action(action, current_price, before_price, number_of_orders)

        elif action == 1:  # Sell stock(-1)
            reward = self._sell_action(action, current_price, before_price, number_of_orders)

        else:
            reward = self._none_action()

        self.holding_price = current_price * self.holdings
        self.reward.append(reward)

    def _get_result_bal(self):
        if self.bal != self.init_bal:
            if self.holding_price != 0:
                self.result.append(self.bal + self.holding_price)

            else:
                self.result.append(self.bal)

        else:
            self.result.append(self.bal)

    def _get_profit_rate(self):
        result = pd.Series(copy.deepcopy(self.result))
        self.profit_rate = np.round(((result.diff().fillna(0) / result.shift().fillna(0)) * 100).fillna(0), 4)

    def step(self, action):
        done = (self.current_step == len(self.state) - 1)

        if done:
            self.reward.append(1)
            print(f'Step: {list(self.state.index)[self.current_step]}')
            obs_state = None
            obs_n_state = None

            self._get_profit_rate()
            self._draw_return_graph()
            self._draw_result_graph()
            self._draw_action_rate()

            print(f"balance is {self.result[-1]}")
            print(f"rest stock is {self.holdings}")
            print(f"profit rate is {np.round(self._get_result_profit(), 5)}%")

        else:
            self._take_action(action)
            self.action_list.append(action)
            self._get_result_bal()
            self.current_step += 1
            obs_state, obs_n_state = self._next_observation()

        if self.train:
            if self.reward[-1] < 0:
                self.reward[-1] = self.reward[-1] * self.nrm

        return obs_state, obs_n_state, self.reward[-1], done

    def _draw_return_graph(self):
        plt.rcParams["font.family"] = 'NanumGothic'
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.rc('axes', unicode_minus=False)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.plot(pd.to_datetime(self.state.index), self.reward)

        if not self.train:
            plt.title(f"{list(self.state.index)[0]} ~ {list(self.state.index)[self.current_step]}"
                      f", profit_rate : {self._get_result_profit()}, target : {self.target}, time: {self.time}")
            plt.savefig(self.dir_path + '/' +
                        f"{list(self.state.index)[0]} ~ {list(self.state.index)[self.current_step]}"
                        f", profit_rate : {self._get_result_profit()}, target : {self.target}, time: {self.time}.png")

        else:
            plt.title(f"train reward graph")

        plt.xlabel('date')
        plt.ylabel('Reward')
        plt.tight_layout()
        plt.show()
        plt.close()

    def _draw_profit_graph(self, ax2):
        open = copy.deepcopy(self.open)
        target_rate = np.round(((open.diff().fillna(0) / open.shift().fillna(0)) * 100).fillna(0), 4)
        target_rate.index = open.index
        ax2.plot(pd.to_datetime(open.index), target_rate, label=f"{self.target} profit rate[EW]")
        ax2.plot(pd.to_datetime(open.index), self.profit_rate.values, label=f"RL profit rate")

        return ax2

    def _get_result_profit(self):
        profit_rate = np.round(((self.result[-1] / self.init_bal) - 1) * 100, 4)

        return profit_rate

    def _draw_sharpe_ratio(self, ax):
        open = copy.deepcopy(self.open)

        if self.train:
            annualization_factor = 252 * 3

        else:
            annualization_factor = 252 * 2

        target_rate = np.round(((open.diff().fillna(0) / open.shift().fillna(0)) * 100).fillna(0), 4)
        target_rate.index = open.index
        target_sharpe_ratio = ((target_rate.expanding().mean()/target_rate.expanding().std()) * np.sqrt(annualization_factor)).fillna(0)
        rl_sharpe_ratio = (self.profit_rate.expanding().mean()/self.profit_rate.expanding().std() * np.sqrt(annualization_factor)).fillna(0)

        print(f"sharpe ratio is {np.round(rl_sharpe_ratio.values[-1],4)}")

        ax.plot(pd.to_datetime(open.index),
                target_sharpe_ratio,
                label=f"{self.target} sharpe ratio[EW] = {np.round(target_sharpe_ratio.values[-1],4)}")

        ax.plot(pd.to_datetime(open.index),
                rl_sharpe_ratio,
                label=f"RL sharpe ratio = {np.round(rl_sharpe_ratio.values[-1],4)}")

        ax.legend()
        ax.set_xlabel('date')
        ax.set_ylabel('sharpe ratio')

        return ax

    def _draw_vwr(self, ax):
        open = copy.deepcopy(self.open)
        ew_vwr = self._get_VWR(prices=open)
        rl_vwr = self._get_VWR(prices=pd.Series(self.result))
        result_dict = {f"{self.target} VWR[EW]" : np.round(ew_vwr['VWR'], 4),
                    f"RL VWR" : np.round(rl_vwr['VWR'], 4)}

        x = np.arange(2)
        p1 = ax.bar(x, result_dict.values())
        ax.legend()
        ax.bar_label(p1, label_type='center')
        ax.set_xticks(x, result_dict.keys())
        ax.set_ylabel('VWR')

        return ax

    def _get_VWR(self, prices, MAV=0.5, TAU=2):
        """
        Compute Variability Weighted Return (VWR)
        참고) https://www.crystalbull.com/sharpe-ratio-better-with-log-returns

        :param pd.Series prices: 가격
        :param float MAV: Maximum Acceptable Variability
        :param float TAU: 가격변동성
        :return: VWR 관련값
        :rtype: dict
        """
        ### 0. Alias
        if self.train:
            annualization_factor = 252 * 3

        else:
            annualization_factor = 252 * 2

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

    def _draw_action(self, ax):
        action_list = np.array(self.action_list)

        date = pd.to_datetime(self.open.index)
        open = np.array(self.open)

        buy_date = pd.to_datetime(date[np.where(action_list == 2)])
        buy = open[np.where(action_list == 2)]

        sell_date = pd.to_datetime(date[np.where(action_list == 1)])
        sell = open[np.where(action_list == 1)]

        ax.plot(date, open, label="Open price")
        ax.plot(buy_date, buy, '^', color='r', label="BUY", markersize=4)
        ax.plot(sell_date, sell, 'v', color='b', label="SELL", markersize=4)

        ax.legend()
        ax.set_title(f'price-action graph : {self.target}, profit_rate : {self._get_result_profit()}')
        ax.set_xlabel('date')
        ax.set_ylabel('action')

        return ax

    def _draw_action_rate(self):
        tur = copy.deepcopy(self.action_list)
        x = np.arange(3)
        buy_rate = round(tur.count(2) / len(tur), 4) * 100
        sell_rate = round(tur.count(1) / len(tur), 4) * 100
        none_rate = round(tur.count(0) / len(tur), 4) * 100
        action_dict = {"buy": buy_rate, "sell": sell_rate, "none": none_rate}

        p1 = plt.bar(x, action_dict.values())
        plt.legend()
        plt.bar_label(p1, label_type='center')
        plt.xticks(x, action_dict.keys())
        plt.ylabel(f'action rate(%), time = {self.time}.png')
        plt.title(f"{self.target}_action_rate.png")

        if not self.train:
            plt.savefig(self.dir_path + '/' + f'{self.target}_action_rate.png, time = {self.time}.png')

        plt.tight_layout()
        plt.show()
        plt.close()

    def _draw_result_graph(self):
        plt.rcParams["font.family"] = 'NanumGothic'
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        self._draw_action(ax1)
        self._draw_sharpe_ratio(ax2)
        self._draw_vwr(ax3)

        if not self.train:
            plt.savefig(self.dir_path + '/' + f"test action, target = {self.target}, time = {self.time}.png")

        plt.tight_layout()
        plt.show()
        plt.close()

    def reset(self):
        self.current_step = 0
        self.reward = []
        self.result = [self.init_bal]
        self.holdings = 0
        self.holding_price = 0
        self.action_list.clear()
        self.trading_price = deque(maxlen=2)
        self.profit_rate = pd.Series()
        self.bal = self.init_bal

        return self._next_observation()

    def render(self, mode='human', close=False):
        if self.current_step % 50 == 0:
            print(f'Step: {list(self.state.index)[self.current_step]}')
            print(f'Reward: {np.round(self.reward[-1], 3)}')
