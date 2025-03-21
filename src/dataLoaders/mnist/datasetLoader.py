import sys
import os
import importlib.util
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow_federated as tff
import json

class DatasetLoader():
    def load_dataset(self, clientid) -> tuple:  # Accetta id come parametro
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        train_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[clientid])
        test_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[clientid])

        def get_x_y_set_reshaped(dataset):
            """
            Reshape dataset in order to give it to the Dense layers
            """
            x_set = np.empty((0, 28, 28))
            y_set = np.empty(0)

            for element in dataset.as_numpy_iterator():
                img = element['pixels']
                label = element['label']

                x_set = np.append(x_set, [img], axis=0)
                y_set = np.append(y_set, label)

            # # reshape data from (value, 28, 28) to (value, 784)
            # x_set_reshaped = x_set.reshape((x_set.shape[0], -1))
            # labels one-hot encoding
            y_one_hot = keras.utils.to_categorical(y_set, 10)

            return x_set, y_one_hot

        x_train, y_train = get_x_y_set_reshaped(train_dataset)
        x_test, y_test = get_x_y_set_reshaped(test_dataset)

        return x_train, x_test, y_train, y_test