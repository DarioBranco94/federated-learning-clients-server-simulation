import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dir_path)
from src.TCPClient import TCPClient
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow_federated as tff


class Client(TCPClient):

    def get_num_classes(self) -> int:
        return 10

    def get_batch_size(self) -> int:
        return 32

    def get_train_epochs(self) -> int:
        return 40

    def get_loss_function(self):
        return keras.losses.CategoricalCrossentropy()

    def get_metric(self):
        return keras.metrics.CategoricalAccuracy()

    def get_skeleton_model(self) -> keras.Model:
        return keras.models.Sequential([
            keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(28, 28)),
            keras.layers.AvgPool1D(strides=2),
            keras.layers.Conv1D(filters=48, kernel_size=5, padding='valid', activation='relu'),
            keras.layers.AvgPool1D(strides=2),
            keras.layers.Flatten(),
            keras.layers.Dense(160, activation='relu'),
            keras.layers.Dense(84, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    def load_dataset(self) -> tuple:
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        train_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[self.id])
        test_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[self.id])

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

    def get_optimizer(self):
        return keras.optimizers.Adam(learning_rate=0.002)


if __name__ == "__main__":
    # get arguments from the console
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int, help='Client ID')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server hostname')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    args = parser.parse_args()

    server_address = (args.host, args.port)

    # Create client
    client = Client(server_address, args.id)
    client.enable_op_determinism()
    client.run()
