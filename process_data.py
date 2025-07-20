import random
import time
import csv
import requests
import numpy as np
import random
from random import randint
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import pickle
import joblib
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle
import joblib





class DataProcessor:
    def __init__(self):
        pass

    def import_data(self):
        
        list_of_yearly_data = []
        complete_players = []
        # career_players = []   # This will hold the career stats of players 

        for i in range(20, 25):
            
            year_string = f"20{i}-{i+1}"

            file_path = f"selenium_data/biweekly/player_data_20{i}-{i+1}_biweekly.csv"

            temp_dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0}

            total_yearly_stats = []
            
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                reader.__next__()  # Skip the header row
                for row in reader:
                    new_row = [row[0], ast.literal_eval(row[1]), year_string]
                    length = len(new_row[1][0]) # Finds the length of the list, not the identifier, in the new row list

                    # print(f"new_row = {new_row[1]}, length = {length}")

                    if length == 10:
                        # Add player to the list of complete players
                        complete_players.append(new_row)
                        # If the player has 10 time splits, add the list of that season to that player's career players list
                        
                    
                    for key, value in temp_dict.items():
                        if length >= int(key):
                            temp_dict[key] += 1
                            
                    # temp_dict[length] += 1

                    total_yearly_stats.append(new_row)
                
            print(f"Year: 20{i}-{i+1}")
            for key, value in temp_dict.items():
                print(f"Number of players with at least {key} time-splits: {value}")

            list_of_yearly_data.append(total_yearly_stats)

        return list_of_yearly_data, complete_players
    

    def find_career_players(self, complete_players):
        # This function will find the career players, which are players that have 10 time splits in a season for multiple seasons
        # It will return a list of career players, which is a list of lists, where each list is a player's career stats

        career_players = []
        temp_dict = dict()

        '''print(f"complete_players[0][2]: {complete_players[0][2]}")  # This prints the first player's first time split data list, which is a list of lists with the time split string and the stats
        return'''

        for player_data in complete_players:
            player_name = player_data[0]
            player_identifier = player_data[1][1]

            # print(f"player_name: {player_name}, player_identifier: {player_identifier}")  # Debugging line to see the player name and identifier
            

            if player_name not in temp_dict: # Make new dictionary entry for new players
                temp_dict[player_name] = [[], player_identifier]  # Initialize the player's list with an empty list and their identifier
            
            temp_name_list = [player_name for player_name in temp_dict]  # This is a list of all the players' lists in the temp_dict

            # print(f"temp_name_list: {temp_name_list}")  # Debugging line to see the list of players in the temp_dict
            
            for dict_player_name in temp_name_list:
                if player_identifier == temp_dict[dict_player_name][1]:
                    temp_dict[player_name][0].append(player_data) # Append the player's time split and year to the player's list in the dictionary
                    break  # Break out of the loop once we find the player, to avoid adding the same player multiple times

            # print(f"temp_dict[player_name][0]: {temp_dict[player_name][0]}")  # Debugging line to see the player's time splits and year
            
            '''if player_identifier not in temp_dict[player_name][0]:
                temp_dict[player_name][0].append(player_identifier)'''

            '''temp_dict[player_name].append(player_data[1][0])  # Append the player's time splits to the dictionary
            print(f"temp_dict[player_name]: {temp_dict[player_name]}")  # Debugging line to see the player's time splits and year'''
            # return
        '''for item in temp_dict["Tavian Dunn-Martin"][0]:
            print(f"item: {item}")'''

        # return

        for player_name in temp_dict:
            player_data_list = temp_dict[player_name][0]
            if len(player_data_list) >= 2:  # Check if the player has multiple seasons
                career_players.append([player_name, player_data_list])

            # player_name format in career_players
            # [player_name, [complete_player_data1, complete_player_data2, ...]] ]


        return career_players


    def parse_start_date(self, range_str):
        start = range_str.split('-')[0]  # "1201" from "1201-1214"
        return datetime.strptime(start, "%Y%m%d")

    def align_player(self, raw_splits, prpg, all_splits): 
        split_dict = dict(zip(raw_splits, prpg))
        return [split_dict.get(s, np.nan) for s in all_splits]



        

    def visualize_data(self, total_yearly_stats, complete_players):
        # Define players of interest
        player_names = [
        # "Cooper Flagg",
        # "V.J. Edgecombe",
        # "Dylan Harper",
        # "Ace Bailey",
        # "Kon Knueppel",
        # "LJ Cryer",
        # "Jayden Dawson", 
        # "Tre White", 
        # "Dajuan Harris",
        # "Flory Bidunga",
        # "KJ Adams",
        # "Hunter Dickinson",
        "Zeke Mayo",
        # "Melvin Council",
        # "Bennett Stirtz",
        # "Brandon Slater",
        # "Hunter Dickinson",
        ]
        colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow']

        # Step 1: Build player name → stats dict
        player_data = {}
        for name in player_names:
            
            matching = [row for row in total_yearly_stats if row[0] == name]
            if not matching:
                print(f"No data found for player: {name}")
                return
            if len(matching) > 1:
                print(f"Multiple entries found for player: {name}. Exiting function . . .")
                return
            player_data[name] = matching[0][1][0]

        # Step 2: Get each player's raw time splits and PRPG values
        player_splits = {}
        player_prpg = {}

        for name in player_names:
            stats = player_data[name]
            raw_splits = [stat[1] for stat in stats]            # e.g., "1201-1214"
            prpg_values = [stat[0][1] for stat in stats]        # stat[0][1] = PRPG value
            player_splits[name] = raw_splits
            player_prpg[name] = prpg_values

        # Step 3: Get full sorted list of all unique time splits
        try:
            all_splits = sorted(
                set(sum(player_splits.values(), [])),
                key=self.parse_start_date
            )
        except Exception as e:
            print(f"Error parsing time splits: {e}")
            return

        # Step 4: Align all players' PRPG to the full timeline
        aligned_data = {}
        for name in player_names:
            aligned_data[name] = self.align_player(player_splits[name], player_prpg[name], all_splits)

        # Step 5: Plot
        plt.figure(figsize=(12, 6))
        for name, color in zip(player_names, colors):
            plt.plot(all_splits, aligned_data[name], marker='o', label=name, color=color)

        plt.title("Points Above Replacement Over Time")
        plt.xlabel("Time Split (YYYYMMDD-YYYYMMDD)")
        plt.ylabel("Points Above Replacement")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return
    
    

    def scale_data(self, complete_players, career_players):

        # Create a list with the same format as complete_players, but without the player identifiers
        scaled_complete_data = []
        scaled_career_data = []
        scaler_list = []

        # player_data example:
        # ['player name', [   [  [[time_split data1, time_split data2, ...], 'time_split_string'], ...    ], "player_identifier"]]

        # These variables below are used to find the length of different lists in complete_players, to be used in loops below
        first_player = complete_players[0]
        first_player_stats = first_player[1]
        first_player_time_splits = first_player_stats[0] # strips the player identifier
        first_player_first_time_split_list_with_timesplit_string = first_player_time_splits[0]  # This is the first player's first time split data list, which will be used the loop below
        first_player_first_time_split_list = first_player_first_time_split_list_with_timesplit_string[0]  # This is the first player's first time split data list without the time split string
        # Above variable is being used in the loop below to determine the number of stats to scale
        print(f"first_player_first_time_split_list: {first_player_first_time_split_list}")

 



        # This loop gets all the stats for one stat catagory to find the scaler of the stat
        for i in range (len(first_player_first_time_split_list)): # Use the length of the first player's first time split data list to determine the number of stats
            # get a stat from all players and scale it
            scaler = StandardScaler()
            unscaled_stat_list = []
            for player_data in complete_players:

                for time_split_list in player_data[1][0]:  # This gets the time split data list for each player
                    # time_split_list = [[time_split data1, time_split data2, ...], 'time_split_string']
                    '''print(f"time_split_list: {time_split_list}")
                    print(f"time_split_list[i] type: {type(time_split_list[0])}")
                    print(f"player_data[1][0]: {player_data[1][0]}")'''
                    # return
                    temp_stat_value = float(time_split_list[0][i])  # This gets the i-th stat from each player's time split data list
                    unscaled_stat_list.append(temp_stat_value)  # This gets the i-th stat from each player's time split data list
            
            numpy_stats = np.array(unscaled_stat_list).reshape(-1, 1)  # Reshape for scaling
            scaled_stat_list = scaler.fit_transform(numpy_stats)  # Scale the stats
            scaler_list.append(scaler)  # Store the scaler for later use
            # scaled_complete_data.append([first_player[0], [scaled_stat_list]])


        # We can start here for career_players, as the above code is finding the scalers, which can still be found
        # with the complete_players data.


        # We might just be able to copy and past the below nested loop for career_players, but just add another
        # for loop to loop through the career_players list.

        # This loop scales the stats for each player using the scalers created above


        
        for complete_season in complete_players:


            scaled_player_stats = [complete_season[0], [[],[],[],[],[],[],[],[],[],[]], complete_season[2]]  # This is the new scaled list of the player's stats

            # This loop loops through all of the time splits for a player
            for k in range(len(first_player_time_splits)):
                time_split_list = complete_season[1][0][k]  # This gets the time split data list for the player (includes time split string)
                # print(f"time_split_list: {time_split_list}")
                # return
                
                # This loop loops through all the stats in a time split for a player
                for i in range(len(first_player_first_time_split_list)):
                    unscaled_stat = float(time_split_list[0][i])
                    scaled_stat = scaler_list[i].transform([[unscaled_stat]])
                    scaled_player_stats[1][k].append(scaled_stat)

            '''print(f"scaled_player_stats: {scaled_player_stats}")
            return'''


            '''print(f"complete_season length: {len(complete_season[1][0])}")
            print(f"complete_season: {complete_season[1][0]}")'''


            scaled_complete_data.append(scaled_player_stats)  # Append the scaled player stats to the scaled_complete_data list
            # return

        
        for complete_player in career_players:

            # complete_player is the list of a player's stats for multiple seasons

            scaled_career_partial_data = []  # This will hold the scaled career data for one season for each player

            complete_player_list = complete_player[1]  # This gets the list of players' data for one season

            for complete_season in complete_player_list:

                '''for item in complete_season:
                    print(f"item: {item}\n")
            
                return'''

                scaled_player_stats = [complete_season[0], [[],[],[],[],[],[],[],[],[],[]], complete_season[2]]  # This is the new scaled list of the player's stats

                # This loop loops through all of the time splits for a player
                for k in range(len(first_player_time_splits)):
                    time_split_list = complete_season[1][0][k]  # This gets the time split data list for the player (includes time split string)
                    # print(f"time_split_list: {time_split_list}")
                    # return
                    
                    # This loop loops through all the stats in a time split for a player
                    for i in range(len(first_player_first_time_split_list)):
                        unscaled_stat = float(time_split_list[0][i])
                        scaled_stat = scaler_list[i].transform([[unscaled_stat]])
                        scaled_player_stats[1][k].append(scaled_stat)
            

                scaled_career_partial_data.append(scaled_player_stats)  # Append the scaled player stats to the scaled_career_partial_data list
                
            scaled_career_data.append(scaled_career_partial_data)  # Append the scaled career data for one season for each player to the scaled_career_data list
                # return

        
        
        print(f"scaler_list = {scaler_list}")

        return scaled_complete_data, scaled_career_data, scaler_list



def main():
    processor = DataProcessor()
    yearly_data, complete_players = processor.import_data()


    career_players = processor.find_career_players(complete_players)

    print(f"length of career_players: {len(career_players)}")

    print(f"length of career_players: {len(career_players[0])}")



    '''print(f"type of data: {type(yearly_data)}")
    print(f"first entry in complete_players: {complete_players[0]}")
    print(f"length of complete_players: {len(complete_players)}")'''

    processor.visualize_data(yearly_data[-1], complete_players)  # Pass in the last year in data

    return
    # THIS RETURN IS HERE SO I CAN JUST VISUALIZE THE DATA WHEN I RUN THIS SCRIPT WHITHOUT MAKING MAJOR CHANGES TO THE DATA. IF YOU WANT TO SCALE THE DATA, REMOVE THIS.


    scaled_complete_data, scaled_career_data, scaler_list = processor.scale_data(complete_players, career_players)

    # return

    # scaled_complete_data is the list of individual complete player seasons. scaled_career_data is the list of lists that contain all seasons for an individual player.

    # return

    # seperated_data is in the format:
    # ['player name', [timesplit_data1, timesplit_data2, ...], 'season_year', [y_values]]


    # print(f"length of player_data[1] in final iteration: {len(scaled_seperated_data[0][1][0])}")

    joblib.dump(scaled_complete_data, "scaled_data_and_scalers/scaled_complete_data.joblib")

    joblib.dump(scaled_career_data, "scaled_data_and_scalers/scaled_career_data.joblib")

    joblib.dump(scaler_list, 'scaled_data_and_scalers/scaler_list.pkl')


    print(f"end of main function in process_data.py")
if __name__ == "__main__":
    main()


# Getting the data to work with missing time splits is a challenge. There are a few options to handle this:
# Option 1: Only train model on players with correct amount of time splits. 
# Option 2: Replace missing time splits with the average of the previous and next time split.
# Option 3: Use a more complex model that can handle missing data.


# Next Steps:
# 1. Transform complete player data into a format suitable for training.
# 1a. Create a function that finds the length of rise or fall from split to split for each player (like the stock market video). 
# 2. Save and export to model file. 
