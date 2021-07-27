import pandas as pd
from copy import deepcopy
from download import download
import os
from random import uniform, choice
from matplotlib import pyplot as plt
class Trader:
    def __init__(self,symbol,episodes=500):

        self.symbol = symbol

        # setting up file structure
        folder_name = f"data/{self.symbol}/"
        print(os.listdir())
        if not os.path.isdir(folder_name): download(self.symbol)
        # getting data from the files
        dset = pd.read_csv(folder_name +"hourly.csv").iloc[::-1].iloc[:, 1:]
        dset.index = range(0, len(dset))

        ema20 = pd.read_csv(folder_name+"20EMA.csv").iloc[::-1].iloc[:, 1:]
        ema20.index = range(0, len(ema20))

        ema50 = pd.read_csv(folder_name+"50EMA.csv").iloc[::-1].iloc[:, 1:]
        ema50.index = range(0, len(ema50))

        sma200 = pd.read_csv(folder_name+"200SMA.csv").iloc[::-1].iloc[:, 1:]
        sma200.index = range(0, len(sma200))

        RSI = pd.read_csv(folder_name+"RSI.csv").iloc[::-1].iloc[:, 1:]
        RSI.index = range(0, len(RSI))

        stoch = pd.read_csv(folder_name+"STOCH.csv").iloc[::-1].iloc[:, 1:]
        stoch.index = range(0, len(stoch))

        bbands = pd.read_csv(folder_name+"bbands.csv").iloc[::-1].iloc[:, 1:]
        bbands.index = range(0, len(bbands))

        VWAP = pd.read_csv(folder_name+"vwap.csv").iloc[::-1].iloc[:, 1:]
        VWAP.index = range(0, len(VWAP))

        sets = [dset, ema20, ema50, sma200, RSI, bbands, VWAP]
        sets = [len(x) for x in sets]
        min_idx = min(sets)

        dset = dset.iloc[:min_idx, :]
        ema20 = ema20.iloc[:min_idx, :].values
        ema50 = ema50.iloc[:min_idx, :].values
        sma200 = sma200.iloc[:min_idx, :].values
        RSI = RSI.iloc[:min_idx, :].values
        stoch = stoch.iloc[:min_idx, :].values
        bbands = bbands.iloc[:min_idx, :].values
        VWAP = VWAP.iloc[:min_idx, :].values

        # declaring feature-arrays:

        cab_20, cab_50, cab_200 = [], [], []  # close above or below ema (20,50,200)
        ab_20_50, ab_50_200 = [], []  # ema 20 above or below ema 50, ema 50 above or below sma200
        RSI_ab_80, RSI_be_20 = [], [] # RSI above 80 or RSI below 20
        k_ab_d = [] # slow-k above slow-d

        c_25_l, c_75_h = [], [] # close 25% away from low, close 75% towards high

        c_75_upperband, c_25_lowerband = [], [] # close 75% towards upperband, close 25% towards lower BollingerBand
        v_ab_c = [] # VWAP above or below close.


        for index, row in dset.iterrows(): # go through every row of data and append 0/1s to the feature-arrays above
            close = float(row["close"])
            high = float(row['high'])
            low = float(row['low'])

            lowerband = bbands[index][0]
            upperband = bbands[index][2]

            vwap = VWAP[index]


            e20 = float(ema20[index])
            e50 = float(ema50[index])
            e200 = float(sma200[index])
            rsi = float(RSI[index])
            d = float(stoch[index][0])
            k = float(stoch[index][1])
            cab_20.append(int(close > e20))
            cab_50.append(int(close > e20))
            cab_200.append(int(close > e20))

            ab_20_50.append(int(e20 > e50))
            ab_50_200.append(int(e50 > e200))
            RSI_ab_80.append(int(rsi > 80))
            RSI_be_20.append(int(rsi < 20))
            k_ab_d.append(int(k > d))
            try:
                travel = (close - low) / (high - low)  # perentage that the price is between high and low
                c_75_h.append(int(travel > 0.75))  # if close is 75% or more towards the high
                c_25_l.append(int(travel < 0.25))  # if close is 25% or less towards the low
            except ZeroDivisionError:
                c_75_h.append(1)
                c_25_l.append(1)
            dist_bet_band = (close - lowerband) / (upperband - lowerband)
            c_75_upperband.append(int(dist_bet_band > 0.75))
            c_25_lowerband.append(int(dist_bet_band > 0.25))

            v_ab_c.append(int(vwap > close))
        dset = dset.iloc[:, 3:4]
        dset["close > ema20"] = cab_20
        dset["close > ema50"] = cab_50
        dset["close > sma200"] = cab_200
        dset["ema20 > ema50"] = ab_20_50
        dset["ema50 > sma200"] = ab_50_200
        dset["RSI < 20"] = RSI_be_20
        dset["RSI > 80"] = RSI_ab_80
        dset["k > d"] = k_ab_d
        dset["close 75% of candle high"] = c_75_h
        dset["close 25% away from candle low"] = c_25_l
        dset["close 75% to upper bollinger band"] = c_75_upperband
        dset["close 25% to lower bollinger band"] = c_25_lowerband
        self.closes = list(dset["close"])
        del dset["close"]  # leaves us only with above/below information.

        # getting average differences:
        diffs = [abs(t - s) for s, t in zip(self.closes, self.closes[1:])]
        avg_diff = sum(diffs) / len(diffs)

        # create the statespace:
        self.environment = []

        # create action space:
        self.action_space = [0, 1, 2, 3]
        # 0 - do nothing (i.e long stays long; short stays short)
        # 1 - Buy
        # 2 - short
        # 3 - close pos

        for index, row in dset.iterrows():
            # here the current position is given as no position
            # because of this only three actions are available (do nothing, buy, short)
            # not close bc obviously you can't close a position that isn't open
            self.environment.append(list(row))

        self.Q = pd.DataFrame(columns=["Do Nothing", "Buy", "Short", "Close"])

        # hyperparameters:
        epsilon = 0.2  # explore/exploit rate. i.e explore 20% of time
        alpha = 0.5  # alpha is the learning rate. I.e q(s[t],a[t]) = q(s[t-1], a[t-1]) + a * new_val

        # measures
        explore_count = 0  # number of times the agent explored the environment (random and sequential (i.e pickign the first value == 0))
        exploit_count = 0  # number of times the agent used previously learnt Q-values
        rand_count = 0  # number of times agent randomly explored
        exploit_balances = []  # balances acheived whilst exploiting
        end_balances = []  # balances acheived any method
        prev_seen = 0  # counts number of states the agent has previously encountered. the lower this number is the better
        # this is because a lower number of previously encountered states means that the agent is being given a detailed description of the state
        # in other words the agent is viewing a high resolution image if it experiences fewer of the same states.
        # however if this number is too low it will be very difficult for the agent to take any decisions at all since all the states will be
        # different and the agent will have no previous experience to use

        for i in range(0, episodes):  # go through the episode episodes-times
            balance = 250  # beginning balance

            state_counter = 0  # s_idx
            done = False  # episode done
            state = self.environment[0] # beginning state will always be the first state in environment
            if str(state) in self.Q.index:  # if the state has already been encountered and is listed
                actions = list(self.Q.loc[str(state)]) # action-space is the listed action space in Q-table
            else: # if this is the first encounter
                actions = 4*[0] # action space is set to 0,0,0,0
                state.append(0)  # assume no position at start of day
            while not done: # until the end of the episode
                try:
                    action = None
                    # 20% of the time + when there are no prepared actions do exploration

                    # indexes of the action_spaces represent the actions
                    # the items themselves are the values associated with the action
                    exploiting = False
                    if any(action == 0 for action in actions):  # i.e if there are any unexplored actions
                        found = False
                        while not found and any(action == 0 for action in actions):
                            action = actions.index(min(a for a in actions if a >= 0))
                            if action != self.get_invalid_action(state[-1]) or action < 0:
                                found = True
                            else:
                                actions[action] = -1
                        if found:
                            explore_count += 1
                    if all(action != 0 for action in actions):  # i.e if all action values populated.
                        if uniform(0, 1) < epsilon:  # i.e 20% of time
                            found = False
                            while not found:
                                action = choice(range(0, 3))
                                if action != self.get_invalid_action(state[-1]):
                                    found = True
                                else:
                                    actions[action] = -1
                            rand_count += 1
                        else:  # exploit
                            action = actions.index(max(actions))
                            exploiting = True
                            exploit_count += 1
                    prev_actions = actions  # this will hold the actions array just taken so that the new Q Value can be added
                    prev_state = state
                    reward = self.r(state, state_counter, action)
                    if exploiting:
                        shares = int(balance / self.closes[state_counter])
                        balance += reward * shares
                    else:
                        balance += reward
                    state, actions, state_counter = self.step(state_counter, state, action)
                    new_val = reward + max(actions)
                    if prev_actions[action] == 0:
                        prev_actions[action] = new_val
                    else:
                        prev_actions[action] += alpha * (new_val - prev_actions[action])
                    self.Q.loc[str(prev_state)] = prev_actions
                    if any(action != 0 for action in actions):
                        prev_seen += 1
                    print(f"For {prev_state=}")
                    print(f"Updated values {prev_actions=}")
                    print(f"{balance=}")
                except IndexError:  # this will occur if self.step() looks for a row in environment which
                    # doesn't exist thus signifying the end of the episode
                    done = True
            if exploiting:
                exploit_balances.append(balance)
            else:
                end_balances.append(balance)
        print(self.Q)
        print(f"Number of times explored: {explore_count}")
        print(f"Number of times exploited: {exploit_count}")
        print(f"Number of times randomly explored: {rand_count}")
        print(f"Number of reencountered situations: {prev_seen}")
        print(
            f"Minimum Balance: {min(end_balances)} | Maximum Balance: {max(end_balances)} | Average Balance: {sum(end_balances) / len(end_balances)}")
        print("When Exploiting")
        print(
            f"Minimum Balance: {min(exploit_balances)} | Maximum Balance: {max(exploit_balances)} | Average Balance: {sum(exploit_balances) / len(exploit_balances)}")
        print(f"position of max balance: {exploit_balances.index(max(exploit_balances))}/{len(exploit_balances)}")
        baseline = 0
        moving_averages = []
        for bal in exploit_balances:
            try:
                total = 0
                for i in range(baseline, baseline + 20):
                    total += exploit_balances[i]
                baseline += 1
                avg = total / 20
                moving_averages.append(avg)
            except IndexError:
                pass

        plt.plot(range(0, len(moving_averages)), moving_averages)
        plt.show()

    def r(self,state, s_idx, action, bal=False):
        """
        works out the reward for taking a given action in a given state
        :param state: current state
        :param s_idx: index of state within environment
        :param action: action to be taken
        :param bal: whether/or not the balance is provided
        :return: integer reward for taking action a in state s
        """
        reward = 0
        curr_close = self.closes[s_idx]  # current close
        next_close = self.closes[s_idx + 1] # next close
        if action == 0:  # i.e hold
            reward += -1
            if state[-1] == 0: # if there is no position
                pass
            elif state[-1] == 1:  # if the position is long
                reward += next_close - curr_close  # reward is the difference between next close and current close
            elif state[-1] == 2:  # if the position is short
                reward += curr_close - next_close
        elif action == 1:  # i.e buy
            reward += next_close - curr_close
        elif action == 2:  # i.e short
            reward += curr_close - next_close
        elif action == 3:  # i.e close pos
            reward += -0.005 * int(state[-1] / curr_close)  # commission estimate.
            if state[-1] == 1:
                reward += curr_close - next_close
            elif state[-1] == 2:  # i.e short position to be closed
                reward += next_close - curr_close

        if bal:
            shares = int(bal / curr_close)
            return reward * shares
        else:
            return reward


    def step(self,s_idx, state, a):
        """
        moves the agent into the next state based on the action taken
        :param s_idx: index of the current state
        :param state: current state
        :param a: action being taken
        :return: new state, new set of actions, new state index
        """
        if a == 0:  # i.e hold
            res_pos = state[-1]
        if a == 1:  # i.e buy
            res_pos = 1
        if a == 2:  # i.e sell
            res_pos = 2
        if a == 3:  # i.e close
            res_pos = 0

        n_state = deepcopy(self.environment)[s_idx + 1]  # gets the next state
        n_state.append(res_pos)
        if str(n_state) in self.Q.index:
            n_actions = list(self.Q.loc[str(n_state)])
            print("Prev. Encountered")
        else:
            n_actions = [0, 0, 0, 0]
        return n_state, n_actions, s_idx + 1

    def get_invalid_action(self,curr_pos):
        if curr_pos == 0:  # if no position
            return 3  # can't close pos
        elif curr_pos == 1:  # if long
            return 1  # can't buy
        elif curr_pos == 2:  # if short
            return 2  # can't sell

Trader("F")