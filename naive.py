import pandas as pd
import tensorflow as tf
from numpy import nan
from pathlib import Path


def risk(estimates, actual): 
    mae = tf.keras.losses.MeanAbsoluteError()
    risk = mae(estimates, actual)
    return risk.numpy()


data_file = Path('labeled_data', 'labeled_data.csv')
data = pd.read_csv(data_file, index_col='Team')
data = data.sort_values(by=['Season', 'P'])
n_seasons = len(data['Season'].unique())
n_teams = 16
n_tot = n_seasons * n_teams

data['EST'] = nan
for i in range(n_seasons):
    data.iloc[i*n_teams + 0,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 1,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 2,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 3,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 4,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 5,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 6,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 7,  data.columns.get_loc('EST')] = 1/64
    data.iloc[i*n_teams + 8,  data.columns.get_loc('EST')] = 1/16
    data.iloc[i*n_teams + 9,  data.columns.get_loc('EST')] = 1/16
    data.iloc[i*n_teams + 10, data.columns.get_loc('EST')] = 1/16
    data.iloc[i*n_teams + 11, data.columns.get_loc('EST')] = 1/16
    data.iloc[i*n_teams + 12, data.columns.get_loc('EST')] = 1/8
    data.iloc[i*n_teams + 13, data.columns.get_loc('EST')] = 1/8
    data.iloc[i*n_teams + 14, data.columns.get_loc('EST')] = 3/16
    data.iloc[i*n_teams + 15, data.columns.get_loc('EST')] = 3/16

output_path = Path('estimates', 'naive.csv')
data.to_csv(output_path)

estimates = data['EST'].to_numpy()
actual = data['PS W%'].to_numpy() 
risk = risk(estimates, actual)

print(f'naive estimate risk: {risk}')

