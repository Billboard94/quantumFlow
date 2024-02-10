import pandas as pd
import json
import numpy as np
import random
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
from tensorflow.data import ExperimentalOps as TFExperimental

# Load the CSV file
df = pd.read_csv('trainingdata_probe.csv')

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
class DataGenerator(TFExperimental.DataSource):
    def __init__(self, dataframe, batch_size=32):
        super().__init__(cardinality=tf.data.experimental.Cardinality.UNBOUNDED, output_types=(tf.float32, tf.float32))
        self.df = dataframe
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df))

    def _generate_elements(self):
        for inputs, outputs in zip(self._map_func(), repeat(None)):
            yield inputs, outputs

    def _map_func(self):
        def map_func(row):
            X = json.loads(row["state"])
            Y = json.loads(row["rho"])
            processed_X = [process_string_to_dict(x["input"]) for x in X]
            processed_Y = [process_string_to_dict(y_str)["output"] for y_str in Y]

            return np.stack(processed_X), np.stack(processed_Y)

        return map_func
    
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
history = model.fit(dataset=data_gen.take(100*tf.data.experimental.AUTOTUNE), epochs=100, verbose=1)