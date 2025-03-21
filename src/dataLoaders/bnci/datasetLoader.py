import sys
import os
import importlib.util
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow_federated as tff
import json
from sklearn.model_selection import train_test_split


class datasetLoader():
    def load_dataset(self, clientid) -> tuple:  # Accetta id come parametro
        folder_path = f"/app/data/input/experiments/BNCI/dataset/{str(int(clientid)+1)}"
        # Check dataset has saved
        if not os.path.isdir(folder_path):
            sys.exit(
                f"Error: '{folder_path}' folder doesn't exist, execute SaveDataset.py first: python3 SaveDataset.py")

        x = np.load(os.path.join(folder_path, "data.npy"))
        y = np.load(os.path.join(folder_path, "labels.npy"))

        print(f"Dataset loaded: {len(x)} items")

        # Add dummy dimension for the Conv net
        x = np.expand_dims(x, axis=-1)

        # Encode labels (0: 'left_hand', 1: 'right_hand')
        y_encoded = np.where(y == 'left_hand', 0, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.25, stratify=y_encoded, random_state=1)

        # labels one-hot encoding
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        return x_train, x_test, y_train, y_test

    def get_optimizer(self):
        return keras.optimizers.Adam(0.002)
