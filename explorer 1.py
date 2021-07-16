import pandas as pd
from random import uniform, choice
from copy import deepcopy
from matplotlib import pyplot as plt
# getting data from the files
symbol = 'F'
dset = pd.read_csv("data/"+symbol+"_hourly.csv").iloc[::-1].iloc[:, 1:]
dset.index = range(0, len(dset))

ema20 = pd.read_csv("data/"+symbol+"_20EMA.csv").iloc[::-1].iloc[:, 1:]
ema20.index = range(0, len(ema20))

ema50 = pd.read_csv("data/"+symbol+"_50EMA.csv").iloc[::-1].iloc[:, 1:]
ema50.index = range(0, len(ema50))

ema200 = pd.read_csv("data/"+symbol+"_200EMA.csv").iloc[::-1].iloc[:, 1:]
ema200.index = range(0, len(ema200))

RSI = pd.read_csv("data/"+symbol+"_RSI.csv").iloc[::-1].iloc[:, 1:]
RSI.index = range(0, len(RSI))

stoch = pd.read_csv("data/"+symbol+"_STOCH.csv").iloc[::-1].iloc[:, 1:]
stoch.index = range(0, len(stoch))

bbands = pd.read_csv("data/"+symbol+"_bbands.csv").iloc[::-1].iloc[:, 1:]
bbands.index = range(0, len(bbands))

VWAP = pd.read_csv("data/" + symbol + "_vwap.csv").iloc[::-1].iloc[:, 1:]
VWAP.index = range(0, len(VWAP))

sets = [dset, ema20, ema50, ema200, RSI, bbands, VWAP]
sets = [len(x) for x in sets]
min_idx = min(sets)

dset = dset.iloc[:min_idx, :]
ema20 = ema20.iloc[:min_idx, :].values
ema50 = ema50.iloc[:min_idx, :].values
ema200 = ema200.iloc[:min_idx, :].values
RSI = RSI.iloc[:min_idx, :].values
stoch = stoch.iloc[:min_idx, :].values
bbands = bbands.iloc[:min_idx, :].values
VWAP = VWAP.iloc[:min_idx, :].values


cab_20, cab_50, cab_200 = [], [], []  # above or below
ab_20_50, ab_50_200 = [], []  # ema 20 above or below ema 50, ema 50 above or below ema200
RSI_ab_80, RSI_be_20 = [], []
k_ab_d = []

c_25_l, c_75_h = [], []

c_75_upperband, c_25_lowerband = [], []
v_ab_c = []

decisions = []  # list of optimal decisions
for index, row in dset.iterrows():
    close = float(row["close"])
    high = float(row['high'])
    low = float(row['low'])

    lowerband = bbands[index][0]
    upperband = bbands[index][2]

    vwap = VWAP[index]
    try:
        fut_close = dset.loc[index+1]["close"]
        decisions.append(int(fut_close > close))
    except KeyError:
        decisions.append("3")
    e20 = float(ema20[index])
    e50 = float(ema50[index])
    e200 = float(ema200[index])
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
        travel  = (close-low)/(high-low) # perentage that the price is between high and low
        c_75_h.append(int(travel > 0.75))  # if close is 75% or more towards the high
        c_25_l.append(int(travel < 0.25))  # if close is 25% or less towards the low
    except ZeroDivisionError:
        c_75_h.append(1)
        c_25_l.append(1)
    dist_bet_band = (close-lowerband)/(upperband-lowerband)
    c_75_upperband.append(int(dist_bet_band > 0.75))
    c_25_lowerband.append(int(dist_bet_band > 0.25))

    v_ab_c.append(int(vwap>close))
dset = dset.iloc[:, 3:4]
dset["close > ema20"] = cab_20
dset["close > ema50"] = cab_50
dset["close > ema200"] = cab_200
dset["ema20 > ema50"] = ab_20_50
dset["ema50 > ema200"] = ab_50_200
dset["RSI < 20"] = RSI_be_20
dset["RSI > 80"] = RSI_ab_80
dset["k > d"] = k_ab_d
dset["close 75% of candle high"] = c_75_h
dset["close 25% away from candle low"] = c_25_l
dset["close 75% to upper bollinger band"] = c_75_upperband
dset["close 25% to lower bollinger band"] = c_25_lowerband
closes = list(dset["close"])
del dset["close"]  # leaves us only with above/below information.

# explanation:
# this zips together two consecutive items in a list and then takes the two items from the tuple and
# populates a list of differences between them.
diffs = [abs(t - s) for s, t in zip(closes, closes[1:])]
avg_diff = sum(diffs)/len(diffs)
# create the statespace:
environment = []

# the state space vectors will have a variable which represents current position
# 0 - no pos; 1 - long; 2 - short.
# create action space:
action_space = [0, 1, 2, 3]
# 0 - do nothing (i.e long stays long; short stays short)
# 1 - Buy
# 2 - short
# 3 - close pos

# now using metrics above:
# create a 'game' where the goal is minimizing -ve distance from init_bal/ maximising +ve distance from it.

for index, row in dset.iterrows():
    # here the current position is given as no position
    # because of this only three actions are available (do nothing, buy, short)
    # not close bc obviously you can't close a position that isn't open
    environment.append(list(row))

indexer = []
[indexer.append(str(state)) for state in environment if str(state) not in indexer]
Q = pd.DataFrame(columns=["Do Nothing", "Buy", "Short", "Close"])


def r(state, s_idx, action, bal=False):
    reward = 0
    curr_close = closes[s_idx]
    next_close = closes[s_idx+1]
    if action == 0:  # i.e hold
        if state[-1] == 0:
            pass
        elif state[-1] == 1:
            reward += next_close-curr_close
        elif state[-1] == 2:
            reward += curr_close-next_close
    elif action == 1:  # i.e buy
        reward += next_close-curr_close
    elif action == 2:  # i.e short
        reward += curr_close-next_close
    elif action == 3:  # i.e close pos
        reward += -0.005 * int(state[-1] / curr_close)  # commission estimate.
        if state[-1] == 1:
            reward += curr_close-next_close
        elif state[-1] == 2:  # i.e short position to be closed
            reward += next_close-curr_close

    if bal:
        shares = int(bal / curr_close)
        return reward*shares
    else:
        return reward


def step(s_idx, state, a):
    if a == 0:  # i.e hold
        res_pos = state[-2]
    if a == 1:  # i.e buy
        res_pos = 1
    if a == 2:  # i.e sell
        res_pos = 2
    if a == 3:  # i.e close
        res_pos = 0

    n_state = deepcopy(environment)[s_idx+1]  # gets the next state
    n_state.append(res_pos)
    if str(n_state) in Q.index:
        n_actions = list(Q.loc[str(n_state)])
        print("Prev. Encountered")
    else:
        n_actions = [0, 0, 0, 0]
    return n_state, n_actions, s_idx+1


def get_invalid_action(curr_pos):
    if curr_pos == 0:  # if no position
        return 3  # can't close pos
    elif curr_pos == 1:  # if long
        return 1  # can't buy
    elif curr_pos == 2:  # if short
        return 2  # can't sell


"""
right now 100 episodes are done
wherein:
    starting from the beginning of the set
    an action is picked either randomly or by choosing the best option
"""

epsilon = 0.2
alpha = 0.5  # alpha is the learning rate

explore_count = 0
exploit_count = 0
rand_count = 0
episodes = 500
exploit_balances = []
end_balances = []
prev_seen = 0
episode_count = 0
for i in range(0, episodes):
    balance = 250

    state_counter = 0
    done = False
    state = environment[0]
    if str(state) in Q.index:
        actions = list(Q.loc[str(state)])
    else:
        actions = [0, 0, 0, 0]
        state.append(0)  # assume no position at start of day
    while not done:
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
                    if action != get_invalid_action(state[-2]) or action < 0:
                        found = True
                    else:
                        actions[action] = -1
                if found:
                    explore_count += 1
            else:  # i.e if all action values populated.
                if uniform(0, 1) < epsilon:  # i.e 20% of time
                    found = False
                    while not found:
                        action = choice(range(0, 3))
                        if action != get_invalid_action(state[-2]):
                            found = True
                    rand_count += 1
                else:  # exploit
                    action = actions.index(max(actions))
                    exploiting = True
                    exploit_count += 1
            prev_actions = actions  # this will hold the actions array just taken so that the new Q Value can be added
            prev_state = state
            reward = r(state, state_counter, action)
            if exploiting:
                shares = int(balance/closes[state_counter])
                balance += reward*shares
            else:
                balance += reward
            state, actions, state_counter = step(state_counter, state, action)
            new_val = reward + max(actions)
            if prev_actions[action] == 0:
                prev_actions[action] = new_val
            else:
                prev_actions[action] += alpha * (new_val-prev_actions[action])
            Q.loc[str(prev_state)] = prev_actions
            if any(action != 0 for action in actions):
                prev_seen += 1
            print(f"For {prev_state=}")
            print(f"Updated values {prev_actions=}")
            print(f"{balance=}")
        except IndexError:
            done = True
    if exploiting:
        exploit_balances.append(balance)
    else:
        end_balances.append(balance)

print(Q)
print(f"Number of times explored: {explore_count}")
print(f"Number of times exploited: {exploit_count}")
print(f"Number of times randomly explored: {rand_count}")
print(f"Number of reencountered situations: {prev_seen}")
print(f"Minimum Balance: {min(end_balances)} | Maximum Balance: {max(end_balances)} | Average Balance: {sum(end_balances)/len(end_balances)}")
print("When Exploiting")
print(f"Minimum Balance: {min(exploit_balances)} | Maximum Balance: {max(exploit_balances)} | Average Balance: {sum(exploit_balances)/len(exploit_balances)}")
baseline = 0
moving_averages = []
for bal in exploit_balances:
    try:
        sum = 0
        for i in range(baseline, baseline+20):
            sum += exploit_balances[i]
        baseline += 1
        avg = sum/20
        moving_averages.append(avg)
    except IndexError:
        pass

plt.plot(range(0, len(moving_averages)), moving_averages)
plt.show()
