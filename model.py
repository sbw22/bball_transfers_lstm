from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed

import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np
import joblib
import ast
import pickle
from sklearn.model_selection import train_test_split



def find_X_and_y(ss_data):
    X = []
    y = []
    player_names = []

    for player_data in ss_data:
        # Assuming player_data is structured as [player_name, [stats]]
        player_name = player_data[0]
        time_splits = player_data[1]


        # Split stats into X (features) and y (target)
        '''for i in range(len(stats) - 1):
            X.append(stats[i])
            y.append(stats[i + 1])
            player_names.append(player_name)'''
        
        temp_stats = list(time_splits)  # Create a copy of stats to avoid modifying the original list
        temp_y = list(player_data[3])

        player_names.append(player_name)
        X.append(temp_stats)
        y.append(temp_y)

        # X and y now contain the stats and the target stat for each player


    return X, y, player_names

def import_stats():
    scaled_seperated_data = joblib.load("scaled_data_and_scalers/scaled_seperated_data.joblib")
    scaler_list = joblib.load("scaled_data_and_scalers/scaler_list.pkl")

    # If it's not iterable or doesn't suapport len(), this will avoid a crash
    '''try:
        print(f"scaled_data length: {len(scaled_seperated_data)}")
    except TypeError as e:
        print(f"TypeError when calling len(): {e}")

    print(f"scaled_data length, type: {len(scaled_seperated_data)}, {type(scaled_seperated_data)}")

    for item in scaled_seperated_data[0]:
        print(f"item: {item}\n")
    print(f"length of item 1 in first item: {len(scaled_seperated_data[0][1][0])}")'''


    # Add function here that makes X and y from the scaled_sperated_data, and makes another list that holds the names of the players in the same order as their stats. 
    X, y, player_names = find_X_and_y(scaled_seperated_data)

    return X, y, player_names, scaler_list


def process_lstm_data(X_raw, y_raw):
    X_formatted = []

    for player_sequence in X_raw:
        # Convert each timestep's list of 16 (1,1) arrays → flat feature vector
        processed_timesteps = []

        for timestep in player_sequence:
            # Flatten each array and extract scalar
            flattened_features = [np.squeeze(f) for f in timestep]
            timestep_vector = np.array(flattened_features)  # shape: (16,)
            processed_timesteps.append(timestep_vector)

        # Now shape is (time_steps, num_features)
        player_matrix = np.stack(processed_timesteps)
        X_formatted.append(player_matrix)

    X_array = np.stack(X_formatted)  # shape: (players, time_steps, features)

    '''y_array = np.array([val[0] if isinstance(val, np.ndarray) else val for val in y_raw])
    y_array = y_array.reshape(-1, 1)'''

    print(f"y_raw[0]: {y_raw[0]}")

    y_array = []

    for row in y_raw:
        
        y_row_flattened = [np.squeeze(f) for f in row]
        y_row_array = np.array(y_row_flattened)  # shape: (16,)
        y_array.append(y_row_array)

    y_array = np.stack(y_array)  # shape: (num_players, 16)

    return X_array, y_array




def main():
    X_raw, y_raw, player_names, scaler_list = import_stats()

    print(f"Number of players: {len(player_names)}")

    X_processed, y_processed = process_lstm_data(X_raw, y_raw)

    print(f"X_processed shape: {X_processed.shape}")
    print(f"y_processed shape: {y_processed.shape}")

if __name__ == "__main__":
    main()