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




'''def find_io(scaled_data):
    pass'''

def import_stats():
    scaled_seperated_data = joblib.load("scaled_data_and_scalers/scaled_seperated_data.joblib")
    scaler_list = joblib.load("scaled_data_and_scalers/scaler_list.pkl")

    # If it's not iterable or doesn't support len(), this will avoid a crash
    try:
        print(f"scaled_data length: {len(scaled_seperated_data)}")
    except TypeError as e:
        print(f"TypeError when calling len(): {e}")

    print(f"scaled_data length, type: {len(scaled_seperated_data)}, {type(scaled_seperated_data)}")

    for item in scaled_seperated_data[0]:
        print(f"item: {item}\n")
    print(f"length of item 1 in first item: {len(scaled_seperated_data[0][1][0])}")

    # Add function here that makes X and y from the scaled_sperated_data, and makes another list that holds the names of the players in the same order as their stats. 


    


def main():
    import_stats()

if __name__ == "__main__":
    main()