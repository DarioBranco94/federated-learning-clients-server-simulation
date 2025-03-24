import sys
import os
import argparse
import subprocess

def run_experiment(dataset_loader, client_id, host, port, num_classes, batch_size, train_epochs, model, loss, metric, optimizer, loss_params, metric_params, optimizer_params, numberOfClients, numberOfRounds,experiment, input_shape, class_names):
    # Modify the base_path to correctly access the examples directory
    #base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', experiment_name))
    base_path = os.path.abspath(os.path.dirname(__file__))
    print(client_id)
    if(client_id != -1):
        client_path = os.path.join(base_path, 'Client.py')
        print((client_path))
        if not os.path.isfile(client_path):
            sys.exit(f"Error: '{experiment}' does not contain valid Client.py or Server.py files.")
        client_process = subprocess.Popen(['python3', client_path, '--id', str(client_id), '--host', str(host), '--port', str(port), '--num_classes', str(num_classes), '--batch_size', str(batch_size), '--train_epochs', str(train_epochs), '--model', str(model), '--dataset_loader', str(dataset_loader), '--loss', str(loss), '--metric', str(metric), '--optimizer', str(optimizer), '--loss_params', str(loss_params), '--metric_params', str(metric_params), '--optimizer_params', str(optimizer_params), '--input_shape', str(input_shape)])	
        client_process.wait()
    else:
        server_path = os.path.join(base_path, 'Server.py')
        if not os.path.isfile(server_path):
            sys.exit(f"Error: '{experiment}' does not contain valid Client.py or Server.py files.")
        server_process = subprocess.Popen(['python3', server_path, '--numberOfClients', str(numberOfClients), '--numberOfRounds', str(numberOfRounds), '--experiment', str(experiment),'--model', str(model),  '--input_shape', str(input_shape), '--class_names', class_names])
        server_process.wait()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, help='Client ID')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server hostname')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--train_epochs', type=int, help='Number of epochs')
    parser.add_argument('--model', type=str, help='Model Name')
    parser.add_argument('--datasetLoader', type=str, help='Experiment name')
    parser.add_argument('--loss', type=str, help='Keras Loss Metric')
    parser.add_argument('--metric', type=str, help='Keras Metric')
    parser.add_argument('--optimizer', type=str, help='Keras Optimizer')
    parser.add_argument('--loss_params', type=str, help='Loss parameters in JSON', default='{}')
    parser.add_argument('--metric_params', type=str, help='Metric parameters in JSON', default='{}')
    parser.add_argument("--optimizer_params", type=str, help="Optimizer parameters in JSON", default='{}')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--numberOfClients', type=int, help='Number of clients')
    parser.add_argument('--numberOfRounds', type=int, help='Number of rounds')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--input_shape', type=str, help='Input shape of the model', default='28,28,1')
    parser.add_argument('--class_names', type=str, help='Class names', default='zero,one,two,three,four,five,six,seven,eight,nine')
    args = parser.parse_args()

    run_experiment(args.datasetLoader, args.id, args.host, args.port, args.num_classes, args.batch_size, args.train_epochs, args.model, args.loss, args.metric, args.optimizer, args.loss_params, args.metric_params, args.optimizer_params, args.numberOfClients, args.numberOfRounds, args.experiment, args.input_shape, args.class_names)