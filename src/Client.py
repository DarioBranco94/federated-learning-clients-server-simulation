import sys
import os
import importlib.util
from TCPClient import TCPClient
import tensorflow.keras as keras
import argparse
import numpy as np
import tensorflow_federated as tff
import json


class Client(TCPClient):
    def __init__(self, server_address, id, num_classes, batch_size, train_epochs, model, dataset_loader, loss , metric , optimizer , loss_params, metric_params, optimizer_params):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.model_instance = self.load_model_class(model)()
        self.get_skeleton_model = self.model_instance.get_skeleton_model
        self.load_dataset = self.load_dataset_function(dataset_loader)

        self.loss_function = self.build_loss_function(loss, loss_params)
        self.metric_function = self.build_metric(metric, metric_params)
        self.optimizer_function = self.build_optimizer(optimizer, optimizer_params)
        super().__init__(server_address, id)

    def load_model_class(self, model_name):
            """ Carica dinamicamente la classe modello da un file nella cartella models."""
            module_path = os.path.join("Inputs/models", f"{model_name}.py")
            spec = importlib.util.spec_from_file_location(model_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, model_name):
                return getattr(module, model_name)
            else:
                raise ValueError(f"Il modulo {model_name} non contiene una classe con nome {model_name}.")
        
    def load_dataset_function(self, dataset_loader_name):
        """ Carica dinamicamente la funzione load_dataset dalla classe DatasetLoader nella cartella Dataset/{dataset_loader_name}."""
        module_path = os.path.join("Inputs/datasets", dataset_loader_name, "datasetLoader.py")
        spec = importlib.util.spec_from_file_location("datasetLoader", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, "DatasetLoader"):
            dataset_loader_class = getattr(module, "DatasetLoader")
            dataset_loader_instance = dataset_loader_class()
            return lambda: dataset_loader_instance.load_dataset(self.id)
        else:
            raise ValueError(f"Il modulo datasetLoader in {dataset_loader_name} non contiene la classe DatasetLoader.")
    def get_num_classes(self) -> int:
        return self.num_classes

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_train_epochs(self) -> int:
        return self.train_epochs

    def build_loss_function(self, loss_name, loss_params):
        loss_class = getattr(keras.losses, loss_name)
        params = json.loads(loss_params) if loss_params else {}
        return loss_class(**params)

    def get_loss_function(self):
        return self.loss_function

    def build_metric(self, metric_name, metric_params):
        metric_class = getattr(keras.metrics, metric_name)
        params = json.loads(metric_params) if metric_params else {}
        return metric_class(**params)

    def get_metric(self):
        return self.metric_function

    def build_optimizer(self, optimizer_name, optimizer_params):
        optimizer_class = getattr(keras.optimizers, optimizer_name)
        params = json.loads(optimizer_params) if optimizer_params else {}
        return optimizer_class(**params)

    def get_optimizer(self):
        return self.optimizer_function

    def get_skeleton_model(self) -> keras.Model:
        return self.model_instance.get_skeleton_model()

    def load_dataset(self) -> tuple:
        return self.model_instance.load_dataset()


if __name__ == "__main__":
    # get arguments from the console
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='Client ID')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server hostname')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--train_epochs', type=int, help='Number of epochs')
    parser.add_argument('--model', type=str, help='Model Name')
    parser.add_argument('--dataset_loader', type=str, help='DatasetName')
    parser.add_argument('--loss', type=str, help='Keras Loss Metric')
    parser.add_argument('--metric', type=str, help='Keras Metric')
    parser.add_argument('--optimizer', type=str, help='Keras Optimizer')
    parser.add_argument('--loss_params', type=str, help='Loss parameters in JSON', default='{}')
    parser.add_argument('--metric_params', type=str, help='Metric parameters in JSON', default='{}')
    parser.add_argument("--optimizer_params", type=str, help="Optimizer parameters in JSON", default='{}')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    args = parser.parse_args()

    server_address = (args.host, args.port)

    # Create client
    client = Client(server_address, args.id, args.num_classes, args.batch_size, args.train_epochs, args.model, args.dataset_loader, args.loss, args.metric, args.optimizer, args.loss_params, args.metric_params, args.optimizer_params)
    client.enable_op_determinism()
    client.run()
