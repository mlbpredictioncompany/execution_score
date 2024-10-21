import pandas as pd
import numpy as np

######

# READ DATA
transitions = pd.read_csv('uninformed_transitions.csv')
win_probability = pd.read_csv('uninformed_win_probability.csv')
print(transitions)
print(win_probability)


# PARSE FIELDS
transitions[['inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'batter_score']] = np.array(transitions['game_state'].apply(lambda x: np.array(x.split('-'))).to_list()).astype(int)
transitions[['next_inning', 'next_outs', 'next_runner_1b', 'next_runner_2b', 'next_runner_3b', 'next_batter_score']] = np.array(transitions['next_state'].apply(lambda x: np.array(x.split('-'))).to_list()).astype(int)
transitions['inning_end'] = np.where(transitions['inning'] != transitions['next_inning'], 1, 0)
print(transitions)


# JOIN TRANSITION PROBABILITIES
win_probability['batter_score'] = np.where(win_probability['half_inning'] == 'top', win_probability['away_score'], win_probability['home_score'])

joined = win_probability.merge(transitions, how='left', on=['inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'batter_score'])
print(joined)


# CLEAN NEXT STATE
joined['opp_half_inning'] = np.where(joined['half_inning'] == 'top', 'bottom', 'top')
joined['next_half_inning'] = np.where(joined['inning_end'] == 1, joined['opp_half_inning'], joined['half_inning'])

joined['next_inning'] = np.where((joined['inning_end'] == 1) & (joined['half_inning'] == 'bottom'), joined['inning'] + 1, joined['inning'])

joined['next_home_score'] = np.where(joined['half_inning'] == 'bottom', joined['next_batter_score'], joined['home_score'])
joined['next_away_score'] = np.where(joined['half_inning'] == 'top', joined['next_batter_score'], joined['away_score'])
print(joined)


# JOIN NEXT STATE WIN PROBABILITIES
next_state = win_probability[['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'home_score', 'away_score', 'away_win', 'home_win', 'tie']].copy()
next_state.columns = ['next_half_inning', 'next_inning', 'next_outs', 'next_runner_1b', 'next_runner_2b', 'next_runner_3b', 'next_home_score', 'next_away_score', 'next_away_win', 'next_home_win', 'next_tie']

joined = joined.merge(next_state, how='left', on=['next_half_inning', 'next_inning', 'next_outs', 'next_runner_1b', 'next_runner_2b', 'next_runner_3b', 'next_home_score', 'next_away_score'])
print(joined)


# CALCULATE NET WIN PROBABILITY
joined['batter_wp'] = np.where(joined['half_inning'] == 'top', joined['away_win'] + 0.5 * joined['tie'], joined['home_win'] + 0.5 * joined['tie'])
joined['next_batter_wp'] = np.where(joined['half_inning'] == 'top', joined['next_away_win'] + 0.5 * joined['next_tie'], joined['next_home_win'] + 0.5 * joined['next_tie'])
joined['net_wp'] = joined['next_batter_wp'] - joined['batter_wp']
print(joined)


# CALCULATE WEIGHTED MEAN
joined['weighted_mean'] = joined['net_wp'] * joined['probability']
joined['weighted_mean'] = joined.groupby('game_state')['weighted_mean'].transform('sum') / joined.groupby('game_state')['probability'].transform('sum')
print(joined)


# CALCULATE WEIGHTED STD
joined['weighted_std'] = (joined['net_wp'] - joined['weighted_mean'])**2 * joined['probability']
joined['weighted_std'] = np.sqrt(joined.groupby('game_state')['weighted_std'].transform('sum') / joined.groupby('game_state')['probability'].transform('sum'))
joined['weighted_var'] = joined['weighted_std']**2
print(joined)


# CALCULATE Z-SCORE
joined['z_score'] = (joined['net_wp'] - joined['weighted_mean']) / joined['weighted_std']
print(joined)


