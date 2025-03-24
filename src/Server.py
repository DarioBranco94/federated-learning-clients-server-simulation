import importlib
import sys
import os
from AggregationAlgorithm import FedAvg, FedAvgMomentum, FedAdam, FedSGD, FedMiddleAvg
from TCPServer import TCPServer
import tensorflow.keras as keras
import argparse
import inspect


class Server(TCPServer):
    def __init__(self, server_address, number_clients, number_rounds, experiment_name, save_weights_path, model,  input_shape, class_names):
        self.input_shape = tuple(map(int, args.input_shape.split(",")))
        self.parsed_class_names = args.class_names.split(",")
        self.model_instance = self.load_model_class(model)()

        super().__init__(server_address, number_clients, number_rounds, experiment_name, save_weights_path)

    def load_model_class(self, model_name):
        """Carica dinamicamente l'unica classe presente in un file Python specificato da model_name."""

        # Converte il nome del modulo in percorso file
        model_sanitized = model_name.replace(".", os.sep)
        module_path = f"{model_sanitized}.py"

        # Verifica che il file esista
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Il file {module_path} non esiste.")

        # Carica il modulo dinamicamente
        spec = importlib.util.spec_from_file_location("loaded_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Trova tutte le classi definite direttamente in questo modulo
        classes = [cls for _, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__]

        # Verifica che ci sia una sola classe definita
        if len(classes) != 1:
            raise ValueError(f"Il modulo {model_name} deve contenere esattamente UNA classe, trovate {len(classes)}.")

        # Restituisce la classe trovata
        return classes[0]


    def get_skeleton_model(self) -> keras.Model:
        return self.model_instance.get_skeleton_model(self.input_shape)
    
    def get_classes_name(self) -> list[str]:
        return self.parsed_class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--numberOfClients', type=int, help='Number of clients')
    parser.add_argument('--numberOfRounds', type=int, help='Number of rounds')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--model', type=str, help='Model Name')
    parser.add_argument('--input_shape', type=str, help='Input shape of the model', default='28,28,1')
    parser.add_argument('--class_names', type=str, help='Class names', default='zero,one,two,three,four,five,six,seven,eight')
    args = parser.parse_args()
    server_address = ('0.0.0.0', 12345)

    # Server creation and execution
    server = Server(server_address, args.numberOfClients, args.numberOfRounds,args.experiment, None, args.model, args.input_shape, args.class_names)
    server.set_aggregation_algorithm(FedAvg())
    # server.set_aggregation_algorithm(FedAdam(beta1=0.5,learning_rate=0.01))
    # server.load_initial_weights("weights/prova.npy")
    server.enable_clients_profiling(False)
    server.enable_evaluations_plots(True)
    server.run()
