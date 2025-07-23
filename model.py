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
from keras.losses import Huber
import numpy as np
import joblib
import ast
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image


class MyGUI:

    def __init__(self, career_player_names, model, target_index, X_test_sequence, y_test_sequence, test_player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled):
        self.career_player_names = career_player_names
        self.model = model
        self.target_index = target_index
        self.X_test_sequence = X_test_sequence
        self.y_test_sequence = y_test_sequence
        self.test_player_name = test_player_name
        self.test_player_year = test_player_year
        self.scaler_list = scaler_list
        self.output_steps = output_steps
        self.full_y_true_scaled = full_y_true_scaled

        self.image_label = None  # Initialize image_label to None
        self.root = tk.Tk()
        self.root.geometry("1200x800")  # Set the window size
        self.root.configure(bg='lightblue')
        self.root.title("Basketball Player Stats Prediction")

        self.ku_font = ("helvetica", 16, "bold")
        self.root.title("Basketball Player Stats Prediction App")
        self.label = tk.Label(self.root, text="Welcome to the Basketball Player Stats Prediction App!", font=self.ku_font, bg='lightblue', fg='crimson')
        self.label.pack(pady=20)

        tk.Label(self.root, text="Type a player:").pack(pady=(10, 0))

        self.lst = career_player_names  # List of player names to populate the combo box
        self.combo_box = ttk.Combobox(self.root, value=self.lst)
        self.combo_box.bind('<KeyRelease>', self.search)  # Bind the KeyRelease event to the search function

        self.combo_box.pack(pady=(10, 10))

        self.enter_button = tk.Button(self.root, text="Enter", command=lambda: self.get_player_chart(self.combo_box.get(), self.lst, self.model, self.target_index, self.X_test_sequence, self.y_test_sequence, self.test_player_name, self.test_player_year, self.scaler_list, self.output_steps, self.full_y_true_scaled))
        self.enter_button.pack(pady=(10, 0))

        self.root.mainloop()  # Start the GUI event loop
    

    def get_player_chart(self, player_name, career_player_names, model, target_index, X_test_sequence, y_test_sequence, test_player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled):
        if player_name not in career_player_names:
            print(f"Player {test_player_name} not found in career player names.")
            return
        else:
            if self.image_label != None:  # Check if image_label is not None
                self.image_label.destroy()  # Clear any previous image label
            
            print(f"got player_name: {player_name}")
            reset_player(model, player_name)  # Reset the player data in the model
            # test_and_evaluate_model(model, target_index, X_test_sequence, y_test_sequence, player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled)
            # Using player_name instead of test_player_name

            image_path = "chart_image/player_chart.png"  # Replace with your image path
            img = Image.open(image_path)
            tk_img = ImageTk.PhotoImage(img)
            self.image_label = tk.Label(self.root, image=tk_img)
            self.image_label.image = tk_img  # Prevent image from being garbage collected
            self.image_label.pack()

            return self.image_label  # Return the image label to be used later if needed

    
    def search(self, event):
        value = event.widget.get()
        if value == '':
            self.combo_box['values'] = self.lst
        else:
            data = []

            for item in self.lst:
                if value.lower() in item.lower():
                    data.append(item)

            self.combo_box['values'] = data

    





def select_player(ss_complete_data, ss_career_data, optional_name=""):

    usable_complete_examples = [
        "Cooper Flagg",
        "Dajuan Harris",
        "Hunter Dickinson",
        "Zeke Mayo",
        "Kon Knueppel",
        "V.J. Edgecombe",
        "Eric Dixon",
        "Johni Broome",
        "Kam Jones",
        "Norchad Omier",
        "Curtis Jones",
        "Milos Uzan",
        "Darius Johnson",
        "Mitch Mascari",
        "Augustas Marciulionis",
        "Alston Mason",
        "Devonte' Graham",
    ]

    target_player = optional_name if len(optional_name) > 0 else "Zeke Mayo"
    target_year = "2024-25"

    # print(f"last item in ss_career_data: {ss_career_data[-1]}")
    

    for player_data in ss_complete_data:
        temp_name = player_data[0]
        temp_year = player_data[2]
        # print(f"temp_name = {temp_name}, temp_year = {temp_year}")
        if temp_name == target_player and temp_year == target_year:
            moving_player_data = ss_complete_data.pop(ss_complete_data.index(player_data))
            ss_complete_data.append(moving_player_data)
            break
    
    for player_data in ss_career_data:

        actual_player_data = player_data[0]
        
        temp_name_ = actual_player_data[0]
        # print(f"temp_name_: {temp_name_}")

        if temp_name_ == target_player:
            moving_player_data = ss_career_data.pop(ss_career_data.index(player_data))
            print(f"moving_player_data[0][0] = {moving_player_data[0][0]}")
            ss_career_data.append(moving_player_data)
            print(f"ss_career_data[0][0] = {ss_career_data[0][0][0]}")
            print(f"ss_careeer_data[-1][0][0] = {ss_career_data[-1][0][0]}")
            print(f"ss_complete_data[-1][0[0] = {ss_complete_data[-1][0]}")
            return ss_complete_data, ss_career_data
    

    return


def find_target_player(complete_scaled_data, career_scaled_data):
    # This function will find the y values for the model, which is the stat that we are trying to predict
    
    # Below average, Ok, Average, Pretty good, Great

    # STAT INDEX
    # 0 = MIN%     # Ok                 * *
    # 1 = PRPG!    # Pretty Good        * * * *
    # 2 = BPM.     # Average            * * *
    # 3 = ORTG.    # Pretty Good        * * * *
    # 4 = USG.     # Below Average      *
    # 5 = EFG.     # Great              * * * * *
    # 6 = TS       # Great              * * * * *
    # 7 = OR.      # Ok                 * *
    # 8 = DR       # Ok                 * *
    # 9 = AST      # Average.           * * *
    # 10 = TO      # Ok                 * *
    # 11 = BLK     # Below Average      *
    # 12 = STL     # Ok                 * *
    # 13 = FTR     # Ok                 * *
    # 14 = 2P      # Great              * * * * *
    # 15 = 3P/100  # Ok                 * *
    # 16 = 3P      # Pretty good        * * * *

    # ***************************************************************************************************************
    target_index = 5  # Change this to the index of the stat you want to predict
    # ***************************************************************************************************************
    # target_index IS VERY IMPORTANT: IT DETERMINES WHICH FEATURE TO PLOT AND UNDO SCALING FOR. 

    for player_data in complete_scaled_data:
        complete_y_list = []
        for time_split in player_data[1]:  # Loop through each player's time split data
            # time_split = [[stat1, stat2, ...], 'time_split_string']

            complete_y_list.append(time_split[target_index])  # Append the stat that we are trying to predict to the y_list
            # time_split.pop(target_index)  # Remove the stat that we are trying to predict from the time split data list
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        player_data.append(complete_y_list)  # Append the y values to the end

    for all_player_data in career_scaled_data:  # LOOK AT THIS LATER. 
        
        for player_data in all_player_data:
            complete_y_list = []
            for time_split in player_data[1]:  # Loop through each player's time split data
                # time_split = [[stat1, stat2, ...], 'time_split_string']

                complete_y_list.append(time_split[target_index])  # Append the stat that we are trying to predict to the y_list
                # time_split.pop(target_index)  # Remove the stat that we are trying to predict from the time split data list
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            player_data.append(complete_y_list)  # Append the y values to the end
            


    return complete_scaled_data, career_scaled_data, target_index  # Return the scaled data with the y values appended to the end of each player's data


def find_X_and_y(ss_complete_data, ss_career_data):

    X_complete = []
    y_complete = []
    X_career = []
    y_career = []
    complete_player_names = []
    career_player_names = []
    complete_player_years = []
    career_player_years = []

    # counter = 0
    for player_data in ss_complete_data:
        
        # Assuming player_data is structured as [player_name, [stats]]
        player_name = player_data[0]
        time_splits = player_data[1]
        player_year = player_data[2]

        
        temp_stats = list(time_splits)  # Create a copy of stats to avoid modifying the original list
        temp_y = list(player_data[3])

        complete_player_names.append(player_name)
        complete_player_years.append(player_year)
        X_complete.append(temp_stats)
        y_complete.append(temp_y)

        # X and y now contain the stats and the target stat for each player

    # print(f"ss_career_data[200][-2][1]: {ss_career_data[280][-2][1]}")  # Debugging line to check the second to last season's stats
    
    '''
    print(f"length of ss_career_data[10][-2][1]: {len(ss_career_data[10][-2][1])}")  # Debugging line to check the second to last season's stats
    for item in ss_career_data[10][-2][1]:
        print(f"item: {item}")
    return'''

    counter = 0
    

    for player_data in ss_career_data:

        '''for item in player_data:
            print(f"item: {item}")
        return'''
        career_raw_stats = []

        '''for item in player_data:
            print(f"item: {item}\n")
        return'''
     
        # Assuming player_data is structured as [player_name, [stats]]
        player_name = player_data[0][0]
        # player_seasons = player_data[1]
        player_year = player_data[-1][2]    # Ed
        career_player_names.append(player_name)
        career_player_years.append(player_year)


        '''temp_player_seasons = list(player_seasons)  # Create a copy of stats to avoid modifying the original listå

        for item in temp_player_seasons:
            print(f"item: {item}")
        
        print(f"temp_player_seasons[-1]: {temp_player_seasons[-1]}")  # Debugging line to check the last season's stats'''

        # print(f"player_data[-2][1]: {player_data[-2][1]}")  # Debugging line to check the second to last season's stats
        # player_data[-2][1] is the second to last season's stats for this player. This list that contains the stats contains 10 time splits (at least for the career data). 


        temp_y = list(player_data[-1][-1]) # Get the last season's y stats

        if temp_y == ['2', '0', '2', '4', '-', '2', '5']:
            print(f'wrong_player_data at index {counter}')
            for item in player_data:
                print(f"item: {item}")  # Debugging line to check player_data contents
            print(f"last season")
            for item in player_data[0]:
                print(f"item = {item}")
            print(f"time splits")
            for item in player_data[0][1]:
                print(f"\nitem = {item}")
            
            return
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        

        '''print(f"temp_y: {temp_y}")  # Debugging line to check temp_y

        if len(player_data) < 2:  # Check if the player has at least 2 seasons of data
            print(f"Skipping {player_name} due to insufficient seasons.")
            continue'''

        for i in range(-2, -1):
            season_stats = list(player_data[i][1])  # Get the stats for the season. Contains 10 time splits (at least for the career data).
            career_raw_stats.extend(season_stats)  # Get all the stats from the player's seasons
    

        # Edit this if statement if the amount of seasons we are training for increases. For example, it should be < 20 if we are training for 2 seasons.
        if len(career_raw_stats) < 10:   # Don't know why career_raw stats would be less than 10 sometimes, will have to figure out later. 
            print(f"Skipping {player_name} due to insufficient stats.")
            print(f"player_data = {player_data}")
            return

        

        X_career.append(career_raw_stats)  # Get the second to last season's stats
        
        # X_career.append(temp_stats)
        y_career.append(temp_y)

        # X and y now contain the stats and the target stat for each player
        counter += 1

    return X_complete, y_complete, complete_player_names, complete_player_years, career_player_names, career_player_years, X_career, y_career



def import_stats(player_name=""):
    scaled_complete_data = joblib.load("scaled_data_and_scalers/scaled_complete_data.joblib")
    scaled_career_data = joblib.load("scaled_data_and_scalers/scaled_career_data.joblib")
    scaler_list = joblib.load("scaled_data_and_scalers/scaler_list.pkl")

    scaled_seperated_complete_data, scaled_seperated_career_data, target_index = find_target_player(scaled_complete_data, scaled_career_data)
    

    try:
        scaled_seperated_complete_data, scaled_seperated_career_data = select_player(scaled_seperated_complete_data, scaled_seperated_career_data, player_name)
    except Exception as e:
        print(f"Error finding target player: {e}")
        return None, None, None, None, None


    # Add function here that makes X and y from the scaled_seperated_complete_data, and makes another list that holds the names of the players in the same order as their stats.
    X_complete, y_complete, complete_player_names, complete_player_years, career_player_names, career_player_years, X_career, y_career = find_X_and_y(scaled_seperated_complete_data, scaled_seperated_career_data)


    return X_complete, y_complete, complete_player_names, complete_player_years, career_player_names, career_player_years, scaler_list, X_career, y_career, target_index


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


    # print(f"y_raw[0]: {y_raw[0]}")

    y_array = []

    for row in y_raw:
        
        y_row_flattened = [np.squeeze(f) for f in row]
        y_row_array = np.array(y_row_flattened)  # shape: (16,)
        y_array.append(y_row_array)
    
    
    y_array = np.stack(y_array)  # shape: (num_players, 16)

    return X_array, y_array

def find_train_test_split(X_complete, y_complete, X_career, y_career, complete_player_names, complete_player_years, career_player_names, career_player_years):
    # split data into train test sets

    # print(f"y_career[0]: {y_career[0]}")

    # Use all sequences for training EXCEPT one
    X_complete_train, y_complete_train = X_complete[:-1], y_complete[:-1]
    X_career_train, y_career_train = X_career[:-1], y_career[:-1]

    # Use the last sequence for testing
    X_full_complete_test_sequence = X_complete[-1]  # shape: (time_steps, features)
    y_full_complete_test_sequence = y_complete[-1]  # shape: (features,) or (time_steps, features)

    X_full_career_test_sequence = X_career[-1]  # shape: (time_steps, features)
    y_full_career_test_sequence = y_career[-1]  # shape: (features,) or (time_steps, features)

    complete_test_player_name = complete_player_names[-1]  # Get the name of the last player
    complete_test_player_year = complete_player_years[-1]  # Get the year of the last player

    career_test_player_name = career_player_names[-1]
    career_test_player_year = career_player_years[-1]

    # We'll determine the split based on output_steps in the main function
    # For now, just return the full sequence and let test_and_evaluate_model handle the split
    return X_complete_train, X_full_complete_test_sequence, y_complete_train, y_full_complete_test_sequence, complete_test_player_name, complete_test_player_year, career_test_player_name, career_test_player_year, X_career_train, y_career_train, X_full_career_test_sequence, y_full_career_test_sequence


def slice_y_to_output_steps(y_data, output_steps):   # I am not passing career y data into this function right now, because I am using the entire last year as the y. 
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

    # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    model = keras.models.Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # Assuming input shape is (time_steps, features)
    model.add(LSTM(64))
    model.add(Dense(int(output_steps)))  # Predict all steps at once (flattened)
    # model.add(keras.layers.Reshape((output_steps, y_train.shape[1])))  # Reshape to (steps, features)

    model.compile(optimizer='adam', loss=Huber(delta=1.0))    # Huber(delta=1.0))
    model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.20, shuffle=True)
    return model





#  def test_and_evaluate_model(model, target_index, X_test_sequence, y_test_sequence, test_player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled):
'''def test_and_evaluate_model(model, target_index, X_full_test_sequence, y_full_test_sequence, test_player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled):
    # Now we handle the train/test split here based on output_steps
    seq_len = X_full_test_sequence.shape[0]
    print(f"in test_and_evaluate_model, seq_len: {seq_len}, output_steps: {output_steps}")
    print(f"X_full_test_sequence shape: {X_full_test_sequence.shape}, y_full_test_sequence shape: {y_full_test_sequence.shape}")
    dont_split_data = False  # This variable is used to determine if we should split the data or not. If career data is used, we do split it.
    # Use the last output_steps for testing, everything before for input
    test_start = seq_len - output_steps
    if test_start < 1:
        dont_split_data = True  # Set this to true if we are using career data, so that we don't split the data into input and output.
    
    if dont_split_data == False:
        X_test_input = X_full_test_sequence[:test_start]  # All but last output_steps
        y_test_true = y_full_test_sequence[-output_steps:]  # Last output_steps values

        X_test_input = np.expand_dims(X_test_input, axis=0)
    else:
        X_test_input = X_full_test_sequence 
        y_test_true = y_full_test_sequence  # Use the entire sequence in the case of career data
        
    
    # Reshape input to match LSTM expected shape: (1, time_steps, features)
    
    print(f"X_test_input shape: {X_test_input.shape}, y_test_true shape: {y_test_true.shape}")
    print(f"Using last {output_steps} values for testing, {test_start} time steps for input")
    
    # Predict
    y_pred = model.predict(X_test_input)  # shape: (1, output_steps)
    
    # The model predicts 'output_steps' future time steps
    print(f"Model prediction shape: {y_pred.shape}")



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
        print(f"Step {start_index + i + 1}: Predicted = {y_pred_unscaled[i][0]:.4f}, True = {y_test_unscaled[i][0]:.4f}, Difference = {abs(y_pred_unscaled[i][0] - y_test_unscaled[i][0]):.4f}")

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
    
    plt.show()'''



def test_and_evaluate_model(model, target_index, X_test_sequence, y_test_sequence, test_player_name, test_player_year, scaler_list, output_steps, full_y_true_scaled):
    
    seq_len = X_test_sequence.shape[0]

    # This condition differentiates the 'complete' and 'career' model scenarios.
    if seq_len > output_steps:
        # --- SCENARIO 1: COMPLETE MODEL (Predicting the end of a sequence) ---
        print("--- Running in 'Complete Model' Mode ---")
        
        # 1. Split the data
        test_start = seq_len - output_steps
        X_test_input = X_test_sequence[:test_start]
        y_test_true_values = y_test_sequence[-output_steps:]
        
        # 2. Add batch dimension and predict
        X_test_input = np.expand_dims(X_test_input, axis=0)
        y_pred = model.predict(X_test_input)

        # 3. Unscale everything for plotting
        y_pred_unscaled = scaler_list[target_index].inverse_transform(y_pred.reshape(-1, 1))
        y_test_true_unscaled = scaler_list[target_index].inverse_transform(y_test_true_values.reshape(-1, 1))
        full_y_true_unscaled = scaler_list[target_index].inverse_transform(np.array(full_y_true_scaled).reshape(-1, 1))

        # 4. Print results
        label_list = ["MIN%", "PRPG!", "BPM", "ORTG", "USG", "EFG", "TS", "OR", "DR", "AST", "TO", "BLK", "STL", "FTR", "2P", "3P/100", "3P"]
        chosen_label = label_list[target_index]
        print(f"\n📈 Predicting last {output_steps} value(s) for: {test_player_name} ({test_player_year})")
        for i in range(len(y_pred_unscaled)):
            print(f"Time Step {i + 1}: Predicted = {y_pred_unscaled[i, 0]:.4f}, True = {y_test_true_unscaled[i, 0]:.4f}, Difference = {abs(y_pred_unscaled[i, 0] - y_test_true_unscaled[i, 0]):.4f}")
        # Calculate and print error metrics
        errors = np.abs(y_pred_unscaled.flatten() - y_test_true_unscaled.flatten())
        print(f"\nError Analysis:")
        print(f"Mean Absolute Error: {np.mean(errors):.4f}")
        print(f"Max Error: {np.max(errors):.4f}")
        print(f"Min Error: {np.min(errors):.4f}")
        # 5. Plot the intra-sequence prediction
        plt.figure(figsize=(12, 6))
        plt.plot(full_y_true_unscaled, label="True Full Sequence", color='gold', marker='o')
        pred_x_indices = range(test_start, seq_len)
        plt.plot(pred_x_indices, y_pred_unscaled, label=f"Predicted Last {output_steps} Values", color='red', linestyle='--', marker='x')
        plt.axvline(x=test_start - 0.5, color='gray', linestyle=':', label="Prediction Starts")
        plt.title(f"{test_player_name}'s Sequence Prediction -- {chosen_label}")
        plt.xlabel("Time Period")
        plt.ylabel(f"Value for {chosen_label}")
        plt.legend()
        plt.grid(True)
        # plt.show()

        plt.savefig("chart_image/player_chart.png", bbox_inches='tight')
        plt.close()  # Close the plot to free memory

    else:
        # --- SCENARIO 2: CAREER MODEL (Predicting a full sequence from another) ---
        print("--- Running in 'Career Model' Mode ---")

        # 1. Use the sequences directly, just add the batch dimension to the input.
        X_test_input = np.expand_dims(X_test_sequence, axis=0)
        y_test_true_values = y_test_sequence

        # 2. Predict
        y_pred = model.predict(X_test_input)
        
        # 3. Unscale for comparison
        y_pred_unscaled = scaler_list[target_index].inverse_transform(y_pred.reshape(-1, 1))
        y_test_true_unscaled = scaler_list[target_index].inverse_transform(y_test_true_values.reshape(-1, 1))
        
        # 4. Print results
        label_list = ["MIN%", "PRPG!", "BPM", "ORTG", "USG", "EFG", "TS", "OR", "DR", "AST", "TO", "BLK", "STL", "FTR", "2P", "3P/100", "3P"]
        chosen_label = label_list[target_index]
        print(f"\n📈 Predicting season {test_player_year} for player: {test_player_name}")
        for i in range(len(y_pred_unscaled)):
            print(f"Time Step {i + 1}: Predicted = {y_pred_unscaled[i, 0]:.4f}, True = {y_test_true_unscaled[i, 0]:.4f}, Difference = {abs(y_pred_unscaled[i, 0] - y_test_true_unscaled[i, 0]):.4f}")
        # Calculate and print error metrics
        errors = np.abs(y_pred_unscaled.flatten() - y_test_true_unscaled.flatten())
        print(f"\nError Analysis:")
        print(f"Mean Absolute Error: {np.mean(errors):.4f}")
        print(f"Max Error: {np.max(errors):.4f}")
        print(f"Min Error: {np.min(errors):.4f}")
        # 5. Plot the full sequence-to-sequence prediction
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_true_unscaled, label="True Values", color='gold', marker='o', linewidth=2)
        plt.plot(y_pred_unscaled, label="Predicted Values", color='red', linestyle='--', marker='x')
        plt.title(f"Season Prediction for {test_player_name} ({test_player_year}) -- Stat: {chosen_label}")
        plt.xlabel("Time Period within Season")
        plt.ylabel(f"Value for {chosen_label}")
        plt.legend()
        plt.grid(True)
        # plt.show()

        plt.savefig("chart_image/player_chart.png", bbox_inches='tight')
        plt.close()  # Close the plot to free memory
    
    

def make_ui(career_player_names, career_model, target_index, X_full_career_test_sequence, y_full_career_test_sequence, career_test_player_name, career_test_player_year, scaler_list, career_output_steps, full_y_career_true_scaled):

    my_gui = MyGUI(career_player_names, career_model, target_index, X_full_career_test_sequence, y_full_career_test_sequence, career_test_player_name, career_test_player_year, scaler_list, career_output_steps, full_y_career_true_scaled)


def reset_player(model, player_name):

    X_complete_raw, y_complete_raw, complete_player_names, complete_player_years, career_player_names, career_player_years, scaler_list, X_career_raw, y_career_raw, target_index = import_stats(player_name)

    X_complete_processed, y_complete_processed = process_lstm_data(X_complete_raw, y_complete_raw)
    X_career_processed, y_career_processed = process_lstm_data(X_career_raw, y_career_raw)


    X_complete_shape = X_complete_processed.shape
    X_career_shape = X_career_processed.shape

    X_complete_train, X_full_complete_test_sequence, y_complete_train, y_full_complete_test_sequence, complete_test_player_name, complete_test_player_year, career_test_player_name, career_test_player_year, X_career_train, y_career_train, X_full_career_test_sequence, y_full_career_test_sequence = find_train_test_split(X_complete_processed, y_complete_processed, X_career_processed, y_career_processed, complete_player_names, complete_player_years, career_player_names, career_player_years)

    # FLEXIBLE PARAMETER: Change this to predict any number of future time steps
    # ***************************************************************************************************************
    complete_output_steps = 3  # Number of future time steps to predict for complete data (e.g., 3 = predict next 3 time points)
    career_output_steps = 10  # Number of future time steps to predict for career data
    # ***************************************************************************************************************

    y_train_sliced = slice_y_to_output_steps(y_complete_train, complete_output_steps)
    full_y_complete_true_scaled = y_complete_processed[-1]  # last player's full y sequence (scaled)
    full_y_career_true_scaled = y_career_processed[-1]  # last player's full career y sequence (scaled)

    test_and_evaluate_model(model, target_index, X_full_career_test_sequence, y_full_career_test_sequence, career_test_player_name, career_test_player_year, scaler_list, career_output_steps, full_y_career_true_scaled)



def main():

    X_complete_raw, y_complete_raw, complete_player_names, complete_player_years, career_player_names, career_player_years, scaler_list, X_career_raw, y_career_raw, target_index = import_stats()

    X_complete_processed, y_complete_processed = process_lstm_data(X_complete_raw, y_complete_raw)
    X_career_processed, y_career_processed = process_lstm_data(X_career_raw, y_career_raw)


    X_complete_shape = X_complete_processed.shape
    X_career_shape = X_career_processed.shape

    X_complete_train, X_full_complete_test_sequence, y_complete_train, y_full_complete_test_sequence, complete_test_player_name, complete_test_player_year, career_test_player_name, career_test_player_year, X_career_train, y_career_train, X_full_career_test_sequence, y_full_career_test_sequence = find_train_test_split(X_complete_processed, y_complete_processed, X_career_processed, y_career_processed, complete_player_names, complete_player_years, career_player_names, career_player_years)

    # FLEXIBLE PARAMETER: Change this to predict any number of future time steps
    # ***************************************************************************************************************
    complete_output_steps = 3  # Number of future time steps to predict for complete data (e.g., 3 = predict next 3 time points)
    career_output_steps = 10  # Number of future time steps to predict for career data
    # ***************************************************************************************************************

    y_train_sliced = slice_y_to_output_steps(y_complete_train, complete_output_steps)
    full_y_complete_true_scaled = y_complete_processed[-1]  # last player's full y sequence (scaled)
    full_y_career_true_scaled = y_career_processed[-1]  # last player's full career y sequence (scaled)

    # return
    # ************************************************************************************************************************************************************************
    # EDIT HERE WHICH MODEL WE ARE TRAINING AND PREDICTING, EITHER THE COMPLETE OR CAREER MODEL

    # complete_model = compile_lstm_model(X_complete_train, y_train_sliced, complete_output_steps)  # Compile and train the LSTM model
    career_model = compile_lstm_model(X_career_train, y_career_train, career_output_steps)

    print(f"\nfull_y_career_true_scaled shape = {full_y_career_true_scaled.shape}\n")
    print(f"\nfull_X_career_true_scaled shape = {X_full_career_test_sequence.shape}\n")

    # Tests complete model
    # test_and_evaluate_model(complete_model, target_index, X_full_complete_test_sequence, y_full_complete_test_sequence, complete_test_player_name, complete_test_player_year, scaler_list, complete_output_steps, full_y_complete_true_scaled)  # Test and evaluate the model
    # Tests career model
    make_ui(career_player_names, career_model, target_index, X_full_career_test_sequence, y_full_career_test_sequence, career_test_player_name, career_test_player_year, scaler_list, career_output_steps, full_y_career_true_scaled)
    # test_and_evaluate_model(career_model, target_index, X_full_career_test_sequence, y_full_career_test_sequence, career_test_player_name, career_test_player_year, scaler_list, career_output_steps, full_y_career_true_scaled)  # Test and evaluate the model

    # ************************************************************************************************************************************************************************
    
    print(f"X_complete_processed shape: {X_complete_processed.shape}")
    print(f"y_complete_processed shape: {y_complete_processed.shape}")


    # OPTIONS FOR NEXT STEPS:
    # 1. Make model that predicts a whole season's worth of stats
    # A. Modify the output layer to predict all time steps at once and adjust the loss function accordingly.
    # B. Change the amount of input data to include the entire season's stats, not just biweekly.
    # C. Change the model architecture to handle longer sequences.



    # 2. Change model input data (X) from pure biweekly stats to the difference between biweekly stats

if __name__ == "__main__":
    main()