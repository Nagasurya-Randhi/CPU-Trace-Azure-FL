import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import flwr as fl
import tensorflow as tf
from model import create_attention_model_GRU, create_dataset, apply_kalman_filter
from config import EPOCHS, WINDOW_SIZE, UNITS, DATA_DIR, SERVER_ADDRESS

class TimeSeriesClient(fl.client.Client):
    def __init__(self, client_id):
        self.client_id = client_id
        self.csv_files = self.load_csv_files(client_id)
        input_shape = (WINDOW_SIZE, 1)
        self.model = create_attention_model_GRU(input_shape, UNITS)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.epochs = EPOCHS

    def load_csv_files(self, client_id):
        with open('file_allocation.json', 'r') as f:
            file_allocation = json.load(f)
        client_files = [os.path.join(DATA_DIR, f) for f in file_allocation[str(client_id)]]
        print(f"Client {client_id} has {len(client_files)} files.")
        return client_files

    def preprocess_data(self, file):
        df = pd.read_csv(file)
        df = df[["CPU usage [%]"]].dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        data = df.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        data = apply_kalman_filter(data)
        return data, scaler

    def train_on_file(self, file):
        data, scaler = self.preprocess_data(file)
        if data is None or len(data) == 0:
            return None, None, 0, 0

        training_size = int(len(data) * 0.70)
        train_data, test_data = data[:training_size], data[training_size:]

        if len(train_data) <= WINDOW_SIZE or len(test_data) <= WINDOW_SIZE:
            print(f"Not enough data in {file} to create sequences with window size {WINDOW_SIZE}")
            return None, None, 0, 0

        X_train, y_train = create_dataset(train_data, WINDOW_SIZE)
        X_test, y_test = create_dataset(test_data, WINDOW_SIZE)

        if X_train.size == 0 or y_train.size == 0 or X_test.size == 0 or y_test.size == 0:
            print(f"Insufficient data after creating sequences from {file}")
            return None, None, 0, 0

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=64,
            verbose=0,
            validation_data=(X_test, y_test)
        )
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]

        return train_loss, val_loss, len(X_train), len(X_test)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        total_loss = 0
        total_val_loss = 0
        total_train_samples = 0
        total_test_samples = 0

        for file in self.csv_files:
            train_loss, val_loss, train_samples, test_samples = self.train_on_file(file)
            if train_samples > 0 and test_samples > 0:
                total_loss += train_loss * train_samples
                total_val_loss += val_loss * test_samples
                total_train_samples += train_samples
                total_test_samples += test_samples

        avg_loss = total_loss / total_train_samples if total_train_samples > 0 else float('inf')
        avg_val_loss = total_val_loss / total_test_samples if total_test_samples > 0 else float('inf')

        print(f"Client {self.client_id}: avg_loss = {avg_loss}, avg_val_loss = {avg_val_loss}")

        return self.model.get_weights(), total_train_samples, {'loss': avg_loss, 'val_loss': avg_val_loss}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        total_loss = 0
        total_samples = 0

        for file in self.csv_files:
            data, _ = self.preprocess_data(file)
            training_size = int(len(data) * 0.70)
            _, test_data = data[:training_size], data[training_size:]

            if len(test_data) <= WINDOW_SIZE:
                print(f"Not enough data in {file} to create sequences with window size {WINDOW_SIZE}")
                continue

            X_test, y_test = create_dataset(test_data, WINDOW_SIZE)

            if X_test.size == 0 or y_test.size == 0:
                print(f"Insufficient data after creating sequences from {file}")
                continue

            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            loss = self.model.evaluate(X_test, y_test, verbose=0)
            total_loss += loss * len(y_test)
            total_samples += len(y_test)

        avg_loss = total_samples and total_loss / total_samples or 0

        return avg_loss, total_samples, {'loss': avg_loss}

if __name__ == "__main__":
    client_id = int(os.getenv("CLIENT_ID", 0))
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=TimeSeriesClient(client_id)
    )

