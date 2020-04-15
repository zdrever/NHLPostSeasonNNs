import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from pathlib import Path

input_cols = [
    'W',
    'L',
    'P',
    'P%',
    'RW',
    'ROW',
    'GF',
    'GA',
    'PP%',
    'PK%',
    'Net PP%',
    'Net PK%',
    'Shots/GP',
    'SA/GP',
    'FOW%',
]

data_file = Path('labeled_data', 'labeled_data.csv')
data = pd.read_csv(data_file, index_col='Team')

n_seasons = len(data['Season'].unique())
n_teams = 16
n_params = len(input_cols)

min_max_scaler = preprocessing.MinMaxScaler()

input_data = data[input_cols].to_numpy()
input_data = min_max_scaler.fit_transform(input_data)
input_data = input_data.reshape(
    (n_seasons, n_teams, n_params))
labels = data['PS W%'].to_numpy().reshape((n_seasons, n_teams, ))

test_choice = np.random.randint(0, input_data.shape[0])
test_input = input_data[test_choice].reshape(1, n_teams, n_params)
test_label = labels[test_choice].reshape(1, n_teams)

X = np.delete(input_data, test_choice, axis=0)
Y = np.delete(labels, test_choice, axis=0)

def train_validate(X, Y):
    best_mae = float('inf')
    best_ks = None
    best_hl_size = None
    best_op = None
    best_model = None
    for hl_size in hidden_layer_sizes: 
        for af in activation_functions:
            for op in optimizers:
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(n_teams, n_params)),
                    tf.keras.layers.Dense(hl_size, activation=af),
                    tf.keras.layers.Dense(n_teams, activation='softmax'),
                ])

                model.compile(
                    optimizer=op,
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error']
                )
                
                total_mae = 0
                for k in range(X.shape[0]):
                    train_input = np.delete(X, k, axis=0)
                    train_labels = np.delete(Y, k, axis=0)
                    val_input = X[k].reshape(1, n_teams, n_params)
                    val_label = Y[k].reshape(1, n_teams)

                    model.fit(train_input, train_labels, epochs=1, batch_size=1, shuffle=True)
                
                    _, mae = model.evaluate(val_input, val_label)
                    total_mae += mae
                
                hold_out_one_mae = total_mae / X.shape[0]
                if hold_out_one_mae < best_mae: 
                    best_mae = mae
                    best_hl_size = hl_size
                    best_af = af
                    best_op = op
                    best_model = model

    return best_mae, best_hl_size, best_af, best_op, best_model

hidden_layer_sizes = [32, 64, 128]
activation_functions = ['linear', 'softmax', 'relu']
optimizers = ['adam', 'sgd']

mae, hl_size, af, op, model = train_validate(X,Y)

_, test_accuracy = model.evaluate(X, Y)

print(f'best validation mae: {mae}')
print(f'best hidden layer size: {hl_size}')
print(f'best activation function: {af}')
print(f'best optimization function: {op}')
print(f'test mae: {test_accuracy}')

