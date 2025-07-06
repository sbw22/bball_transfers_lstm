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



class DataProcessor:
    def __init__(self):
        pass

    def import_data(self):
        file_path = f"selenium_data/biweekly/player_data_2024-25_biweekly.csv"

        temp_dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, "10": 0, "11": 0}


        total_stats = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            reader.__next__()  # Skip the header row
            for row in reader:
                new_row = [row[0], ast.literal_eval(row[1])]
                # print(f"new_row = {new_row}")
                # return
                length = len(new_row[1][0]) # Finds the length of the list, not the identifier, in the new row list

                print(f"new_row = {new_row[1]}, length = {length}")

                if length > 14:
                    print(f"Player with more than 14 time-splits: {new_row[0]} with {length} time-splits")
                # FIGURE OUT THIS LOGIC
                '''if length not in temp_dict:
                    temp_dict[length] = 0'''
                
                for key, value in temp_dict.items():
                    if length >= int(key):
                        temp_dict[key] += 1
                        
                # temp_dict[length] += 1

                total_stats.append(new_row)
            
        print(f"Total temp_dict: {temp_dict}")
        for key, value in temp_dict.items():
            print(f"Number of players with at least {key} time-splits: {value}")
        
            


        return total_stats

    
    def parse_start_date(self, range_str):
        start = range_str.split('-')[0]  # "1201" from "1201-1214"
        return datetime.strptime(start, "%Y%m%d")

    def align_player(self, raw_splits, prpg, all_splits): 
        split_dict = dict(zip(raw_splits, prpg))
        return [split_dict.get(s, np.nan) for s in all_splits]



        

    def visualize_data(self, total_stats):
        # Define players of interest
        player_names = [
        #"Jayden Dawson", 
        #"Tre White", 
        "Dajuan Harris",
        "Flory Bidunga",
        "KJ Adams",
        # "Melvin Council",
        # "Cooper Flagg",
        "Bennett Stirtz",
        ]
        colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow']

        # Step 1: Build player name → stats dict
        player_data = {}
        for name in player_names:
            
            matching = [row for row in total_stats if row[0] == name]
            if not matching:
                print(f"No data found for player: {name}")
                return
            if len(matching) > 1:
                print(f"Multiple entries found for player: {name}. Using the first entry.")
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


def main():
    processor = DataProcessor()
    data = processor.import_data()

    processor.visualize_data(data)

    '''for row in data:
        print(f"Player: {row[0]}, stats = {row[1]}")'''



if __name__ == "__main__":
    main()



# Option 1: Only train model on players with correct amount of time splits. 
# Option 2: Replace missing time splits with the average of the previous and next time split.
# Option 3: Use a more complex model that can handle missing data.