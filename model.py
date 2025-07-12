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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def find_X_and_y(ss_data):
    X = []
    y = []
    player_names = []
    player_years = []

    counter = 0
    for player_data in ss_data:
        
        # Assuming player_data is structured as [player_name, [stats]]
        player_name = player_data[0]
        time_splits = player_data[1]
        player_year = player_data[2]

        
        temp_stats = list(time_splits)  # Create a copy of stats to avoid modifying the original list
        temp_y = list(player_data[3])

        player_names.append(player_name)
        player_years.append(player_year)
        X.append(temp_stats)
        y.append(temp_y)

        # X and y now contain the stats and the target stat for each player


    return X, y, player_names, player_years

def find_target_player(ss_data):
    target_player = "Hunter Dickinson"
    target_year = "2024-25"

    for player_data in ss_data:
        temp_name = player_data[0]
        temp_year = player_data[2]
        if temp_name == target_player and temp_year == target_year:
            moving_player_data = ss_data.pop(ss_data.index(player_data))
            ss_data.append(moving_player_data)
            return ss_data

    return

def import_stats():
    scaled_seperated_data = joblib.load("scaled_data_and_scalers/scaled_seperated_data.joblib")
    scaler_list = joblib.load("scaled_data_and_scalers/scaler_list.pkl")

    try:
        scaled_seperated_data = find_target_player(scaled_seperated_data)
    except Exception as e:
        print(f"Error finding target player: {e}")
        return None, None, None, None, None

    # Add function here that makes X and y from the scaled_sperated_data, and makes another list that holds the names of the players in the same order as their stats. 
    X, y, player_names, player_years = find_X_and_y(scaled_seperated_data)

    return X, y, player_names, player_years, scaler_list


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


    print(f"y_raw[0]: {y_raw[0]}")

    y_array = []

    for row in y_raw:
        
        y_row_flattened = [np.squeeze(f) for f in row]
        y_row_array = np.array(y_row_flattened)  # shape: (16,)
        y_array.append(y_row_array)

    y_array = np.stack(y_array)  # shape: (num_players, 16)

    return X_array, y_array

def find_train_test_split(X, y, player_names, player_years):
    # split data into train test sets

    # Use all sequences for training EXCEPT one
    X_train, y_train = X[:-1], y[:-1]

    # Use the last sequence for testing
    X_full_test_sequence = X[-1]  # shape: (time_steps, features)
    y_full_test_sequence = y[-1]  # shape: (features,) or (time_steps, features)
    test_player_name = player_names[-1]  # Get the name of the last player
    test_player_year = player_years[-1]  # Get the year of the last player

    # We'll determine the split based on output_steps in the main function
    # For now, just return the full sequence and let test_and_evaluate_model handle the split
    return X_train, X_full_test_sequence, y_train, y_full_test_sequence, test_player_name, test_player_year


def slice_y_to_output_steps(y_data, output_steps):
    sliced_y = []
    for row in y_data:
        row = np.array(row)
        target_seq = row[-output_steps:]  # Get last `output_steps` values
        sliced_y.append(target_seq)
    return np.stack(sliced_y)


def compile_lstm_model(X_train, y_train, output_steps):

    '''lstm_input = Input(shape=(10, 16), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split = 0.1)'''

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    model = keras.models.Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # Assuming input shape is (time_steps, features)
    model.add(LSTM(64))
    model.add(Dense(int(output_steps)))  # Predict all steps at once (flattened)
    # model.add(keras.layers.Reshape((output_steps, y_train.shape[1])))  # Reshape to (steps, features)

    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train, y_train, epochs=200, batch_size=32)
    return model






def test_and_evaluate_model(model, X_full_test_sequence, y_full_test_sequence, test_player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled):
    # Now we handle the train/test split here based on output_steps
    seq_len = X_full_test_sequence.shape[0]
    
    # Use the last output_steps for testing, everything before for input
    test_start = seq_len - output_steps
    
    X_test_input = X_full_test_sequence[:test_start]  # All but last output_steps
    y_test_true = y_full_test_sequence[-output_steps:]  # Last output_steps values
    
    # Reshape input to match LSTM expected shape: (1, time_steps, features)
    X_test_input = np.expand_dims(X_test_input, axis=0)
    
    print(f"X_test_input shape: {X_test_input.shape}, y_test_true shape: {y_test_true.shape}")
    print(f"Using last {output_steps} values for testing, {test_start} time steps for input")
    
    # Predict
    y_pred = model.predict(X_test_input)  # shape: (1, output_steps)
    
    # The model predicts 'output_steps' future time steps
    print(f"Model prediction shape: {y_pred.shape}")
    
    # For plotting, we'll use the first feature/scaler
    target_index = 1  # Use selected feature/scaler for plotting
    # target_index IS VERY IMPORTANT: IT DETERMINES WHICH FEATURE TO PLOT AND UNDO SCALING FOR. 
    # target_index MUST MATCH THE VARIABLE stat_index, WHICH WAS DEFINED IN PROCESS_DATA.PY. THESE
    # VARIABLES PICK WHICH FEATURE TO PLOT AND UNDO SCALING FOR.

    # Extract all predicted time steps for the first feature
    y_pred_flat = y_pred.flatten().reshape(-1, 1)  # All predictions as column vector
    y_test_flat = y_test_true.reshape(-1, 1)   # All true test values
    
    # Inverse transform using the first scaler
    y_pred_unscaled = scaler_list[target_index].inverse_transform(y_pred_flat)
    y_test_unscaled = scaler_list[target_index].inverse_transform(y_test_flat)
    
    # Unscale full true y sequence for plotting (using first scaler)
    full_y_true_scaled_feature = np.array(full_y_true_scaled).reshape(-1, 1)
    full_y_true_unscaled = scaler_list[target_index].inverse_transform(full_y_true_scaled_feature)

    # Find start point of prediction - we're predicting the last output_steps of the gold line
    start_index = len(full_y_true_unscaled) - output_steps

    # Build predicted timeline: connect to the value before predictions start
    pred_x_indices = list(range(start_index - 1, start_index + output_steps))
    pred_y_values = [full_y_true_unscaled[start_index - 1, 0]] + list(y_pred_unscaled.flatten())

    label_list = ["MIN%", "PRPG!", "BPM", "ORTG", "USG", "EFG", "TS", "OR", "DR", "AST", "TO", "BLK", "STL", "FTR", "2P", "3P/100", "3P"]
    chosen_label = label_list[target_index]  # Get the label for the chosen feature

    # Print predicted vs true
    print(f"\nPredicting last {output_steps} value(s) of the sequence for player: {test_player_name} in the {test_player_year} season")
    print(f"\nPredicted vs True Values (Unscaled):")
    for i in range(len(y_test_unscaled)):
        print(f"Step {start_index + i}: Predicted = {y_pred_unscaled[i][0]:.4f}, True = {y_test_unscaled[i][0]:.4f}")
    
    # Calculate and print error metrics
    errors = np.abs(y_pred_unscaled.flatten() - y_test_unscaled.flatten())
    print(f"\nError Analysis:")
    print(f"Mean Absolute Error: {np.mean(errors):.4f}")
    print(f"Max Error: {np.max(errors):.4f}")
    print(f"Min Error: {np.min(errors):.4f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(full_y_true_unscaled, label="True Full Sequence", color='gold')
    
    # Plot the prediction line (connection + predictions) with markers only on predictions
    plt.plot(pred_x_indices, pred_y_values, color='red', linestyle='--')
    plt.plot(pred_x_indices[1:], pred_y_values[1:], label=f"Predicted Last {output_steps} Values", 
             color='red', marker='x', linestyle='None')
    
    # plt.axvline(start_index, color='gray', linestyle=':', label="Prediction Starts")
    plt.title(f"{test_player_name}'s Sequence with Last {output_steps} Values Predicted In The {test_player_year} Season -- {chosen_label}")
    plt.xlabel("Time Period")
    plt.ylabel(f"{chosen_label}")
    plt.legend()
    plt.grid(True)
    
    # Custom x-axis labels
    custom_labels = ["Nov 1-14", "Nov 14-Dec 1", "Dec 1-14", "Dec 14-Jan 1", 
                     "Jan 1-14", "Jan 14-Feb 1", "Feb 1-14", "Feb 14-Mar 1", 
                     "Mar 1-14", "Mar 14-Apr 1"]
    
    # Set custom x-axis labels (only if we have enough labels for the data)
    if len(custom_labels) >= len(full_y_true_unscaled):
        plt.xticks(range(len(full_y_true_unscaled)), custom_labels[:len(full_y_true_unscaled)], rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.15)  # Make room for rotated labels
    else:
        plt.tight_layout()  # Only use tight_layout if not using custom labels
    
    plt.show()



def main():
    X_raw, y_raw, player_names, player_years, scaler_list = import_stats()

    print(f"Number of players: {len(player_names)}")

    X_processed, y_processed = process_lstm_data(X_raw, y_raw)

    X_shape = X_processed.shape

    X_train, X_full_test_sequence, y_train, y_full_test_sequence, test_player_name, test_player_year = find_train_test_split(X_processed, y_processed, player_names, player_years)

    # FLEXIBLE PARAMETER: Change this to predict any number of future time steps
    output_steps = 3  # Number of future time steps to predict (e.g., 3 = predict next 3 time points)
    
    y_train_sliced = slice_y_to_output_steps(y_train, output_steps)
    full_y_true_scaled = y_processed[-1]  # last player's full y sequence (scaled)

    model = compile_lstm_model(X_train, y_train_sliced, output_steps)  # Compile and train the LSTM model

    test_and_evaluate_model(model, X_full_test_sequence, y_full_test_sequence, test_player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled)  # Test and evaluate the model

    print(f"X_processed shape: {X_processed.shape}")
    print(f"y_processed shape: {y_processed.shape}")

if __name__ == "__main__":
    main()