import pandas as pd
import json
import numpy as np
import random
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Load the CSV file
df = pd.read_csv('yourfile.csv')

# Define helper functions
def process_string_to_dict(data):
    """Convert a given string containing multiple complex numbers into a dict"""

    def parse_complex_number(value):
        """Parse a complex number from its string representation."""
        match = re.findall(r"(-?\d*\.\d+|\d+)[eE][+-]?\d+j?", value)
        return [complex(x.replace("j", "+0j")) for x in match]

    parsed_numbers = parse_complex_number(data)
    num_samples = int((np.sqrt(4 * len(parsed_numbers) + 1) - 1)/2)
    input_matrix = np.zeros([num_samples, num_samples], dtype="complex")
    output_vector = np.zeros(num_samples, dtype="complex")

    idx = 0
    for i in range(num_samples):
        for j in range(num_samples):
            if idx >= len(parsed_numbers):
                break
            input_matrix[i, j] = parsed_numbers[idx]
            idx += 1
        
        if idx >= len(parsed_numbers):
            break
        output_vector[i] = parsed_numbers[idx]
        idx += 1

    return {'input': input_matrix, 'output': output_vector}

# Custom data generator class
class DataGenerator():
    def __init__(self, dataframe, batch_size=32):
        self.df = dataframe
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df))

    def __iter__(self):
        random.shuffle(self.indexes)
        for start_idx in range(0, len(self.df), self.batch_size):
            stop_idx = min(start_idx + self.batch_size, len(self.df))
            yield self.__getitem__(slice(start_idx, stop_idx))

    def __getitem__(self, slice_obj):
        X = [json.loads(self.df.loc[indices]['state']) for indices in slice_obj]
        y = [json.loads(self.df.loc[indices]['rho']) for indices in slice_obj]

        processed_X = [process_string_to_dict(x['input']) for x in X]
        processed_y = [process_string_to_dict(y_str)['output'] for y_str in y]

        return np.stack(processed_X), np.stack(processed_y)

# Prepare the data generator
data_gen = DataGenerator(df, batch_size=32)

# Model architecture
model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=(None, 1)))
model.add(Dense(units=1, activation='linear'))

# Compile the model
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)

# Train the model
history = model.fit(generator=data_gen, epochs=100, verbose=1)