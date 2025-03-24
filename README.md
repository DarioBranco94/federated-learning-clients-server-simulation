# Federated Learning: clients/server desktop implementation
This is a design of a simple client/server architecture to simulate federated learning involving real nodes, where each node (client) owns its own data. The client and server are written in Python and communicate via sockets using the TCP transport protocol.

## Table of contents

* [Overview](#overview)
* [Libraries](#libraries)
* [Architecture](#architecture)
    + [Server](#server)
        - [Methods](#methods)
            * [Abstract Methods](#abstract-methods)
            * [Public Methods](#public-methods)
    + [Client](#client)
        - [Methods](#methods-1)
            * [Abstract Methods](#abstract-methods-1)
            * [Public Methods](#public-methods-1)
    + [Supported Aggregation Algorithms](#supported-aggregation-algorithms)
    + [Message Exchange](#message-exchange)
    + [Profiling](#profiling)
    + [Limits of the Implementation](#limits-of-the-implementation)
* [Requirements](#requirements)
* [Simulation Mnist Dataset](#simulation-mnist-dataset)
* [Simulation BNCI2014_001 Dataset](#simulation-bnci2014_001-dataset)
* [Distributed Deployment With Kubernetes and Helm](#distributed-deployment-with-kubernetes-and-helm)
    + [Install Docker](#install-docker)
    + [Install Minikube](#install-minikube)
    + [Install Helm](#install-helm)
    + [Building Images](#building-docker-images)
    + [Configure Execution](#configure-execution)
    + [Deploy with Helm](#deploy-with-helm)
    + [Cleanup](#cleanup)

## Overview

This document provides a detailed description of the implementation of our Federated Learning (FL) framework for benchmarking. The framework is designed to facilitate the simulation and performance analysis of various federated algorithms. It consists of a central server and multiple clients, each training a model locally on their data and contributing to the global aggregation without sharing raw data. 
The implementation is done in Python (version 3.10), leveraging its simplicity and extensive libraries for machine learning. Clients and server communicate via sockets using the TCP/IP protocol. 

## Libraries

The following libraries are used to ensure the efficiency and reusability of the code:

- **TensorFlow**: For building and training machine learning models.
- **NumPy**: For mathematical operations and data manipulation.
- **Pickle**: For serializing and deserializing Python objects.
- **Struct**: For converting data formats to and from bytes.
- **Threading**: For creating and managing multiple threads.
- **Socket**: For network communication between clients and the server.
- **Os**: For interacting with the operating system.
- **Trace**: For tracing the number of instructions executed during model training.
- **Resource**: For monitoring system resource usage.
- **Matplotlib**: For creating graphs and visualizations.

## Architecture
![fl_architecture](/images/fl_arc.png)

Here is a representation of the architecture, as shown in the figure. The server awaits the connection of nodes participating in federated learning. Whenever a client connects, the server sends an initialized federated model. Once a certain number of nodes are connected, the federated learning process begins:

1. Clients train their local models on local data using federated weights received from the server.
2. Upon completion of local training, clients send the new weights to the server.
3. The server aggregates the weights from all clients, generating a new model.
4. The server sends the new federated model to the clients.
This sequence is iterated for a predetermined number of rounds, at the end of which, clients send the server accuracy and loss data of the federated model before closing the connection with the server.

## File Structure

The project is organized into several top-level folders, each with a distinct purpose to support simulation, configuration, and execution of federated learning experiments.

-   **data/**

    Contains all datasets and experiment outputs.

    -   **data/input/**   
        Used to load local datasets (if not using TFF datasets). This can contain CSV, image folders, or pre-processed data files.
    -   **data/output/**   
        Stores the outputs of each experiment.
        -   **evaluations/**   
            Contains subfolders named after each experiment. Each folder includes:
            -   Model performance metrics: accuracy, loss, confusion matrix
            -   An XML report with:
                -   The **configuration model** used (neural network structure)
                -   The **deployment profile**, describing the machine where the experiment was executed
                -   Key Performance Indicators (**KPIs**), such as:
                    -   Accuracy
                    -   Loss
                    -   Number of instructions
                    -   RAM usage
                    -   Execution time
        -   **logs/**   
            Stores the log files for both clients and server from the most recent execution.
-   **helm/**

    Contains everything related to the deployment of the Federated Learning system using **Helm** and **Kubernetes**.

    -   **templates/**   
        Helm templates defining Kubernetes resources:
        -   **server-deployment.yaml**:   
        Deployment and Service definition for the central server
        -   **client-job.yaml**:   
        Indexed Job for parallel training on multiple clients with unique IDs
        -   **pv.yaml and pvc.yaml**:   
        Persistent Volumes and Claims to retain:
            -   **Logs**
            -   **Evaluation results**
        These volumes ensure that experiment data persists even after pods are terminated.
    -   **Chart.yaml**   
        Main configuration file for the Helm chart. Defines metadata and dependencies for the FL deployment.
    -   **values.yaml**   
        User-editable configuration file for the experiment.   
        It defines the parameters for launching a Federated Learning session, we will describe in details those parameters in next sections.
        
        The experiment results are saved in `/app/output/evaluations/`, and logs in `/app/output/logs/`.   
        These folders are mounted from Kubernetes volumes and **should not be modified manually**.

-   **src/**

    The core logic of the benchmarking framework. It contains all the codebase to run the Benchmarking software along with user defined classes to allow the usage of user defined models and data.

    -   **models/**   
        Contains all available deep learning models.   
        Each model must be implemented as a single class and must define the method:
    
        `get_skeleton_model(input_shape) -> keras.Model 
         `
    
    -   **dataloaders/**   
        Contains dataset loader classes used to simulate data distribution to clients.   
        Each loader class must implement:
    
        `load_dataset(client_id) -> x_train, x_test, y_train, y_test 
         `
    
        In real federated learning, each client owns its own data. Here, data is **simulated** and partitioned using client_id.

##  Server
The server component is defined by the abstract class [TCPServer](/src/TCPServer.py). The constructor requires four main parameters: 
- `server_address`: The server address.
- `number_clients`: The number of participating clients.
- `number_rounds`: The number of federated learning rounds.
- `experiment_name`: The name of the current experiment, it is also used as the name of the directory in which to save experiment outputs 
- `save_weights_path` (optional): Path to save the federated learning weights. If not specified, the model is not saved.

Abstract class TCPServer is implemented by the [Server](/src/Server.py) Class. The constructor of the Server Class requires also: 
- `model`: The path in which the model class is saved (e.g. src.models.model1). This class must implement get_skeleton_model(). 
- `input_shape`: The input shape of the model (e.g. 253,1). Must be formatted as a comma-separated string, and converted to a tuple at runtime. 
- `class_names`: The classes names 

The ```run()``` function is executed to run the server. 

The server opens a socket at the specified address, in our case, localhost:12345, and three threads are created:

+ A thread, ```thread_client_connections```, listens to accept client connections. It accepts up to a number of connections equal to ```number_clients```. For each connected client, a ```client_thread``` is associated with it to handle communication.
+ A thread, ```thread_fl_algorithms```, manages rounds for federated learning. Once all clients send their weights, it calculates the average of all weights and sends the new model to the clients.
+ A thread, ```thread_final_evaluations```, performs the final evaluation of the learning. When the learning process concludes, it collects the accuracies and losses of each local model and creates graphs.

At this point, the server waits for client connections, and when a connection is initialized, the server sends configuration values an initialized model with weights and biases dependent on ```kernel_initializer``` and ```bias_initializer```, respectively defined in the layers of the Keras model returned by the ```get_skeleton_model()``` function. The currently supported configuration is ```profiling```, which allows enabling or disabling profiling for the node in question.

When all clients are connected, the server waits for the reception of local models, thus starting the learning phase that lasts for a number of rounds defined by the ```number_rounds``` variable. At each round, when the server collects all models, the following sequence of events occurs:

1. Models (weights and biases) are aggregated by taking the average of the models.
2. The server sends the resulting model to all clients.

The learning process concludes when the number of rounds is exhausted, resulting in a final model. The thread responsible for evaluation proceeds to provide graphs that describe the learning progress, including:

+ **Average accuracy of the final model**.
+ **Average loss of the final model**.
+ **Trend of accuracy per round for predicting test samples for each client**.
+ **Trend of loss per round for predicting test samples for each client**.
+ **Trend of average accuracy per round**.
+ **Trend of average loss per round**.
+ **Confusion matrix of the final model (mean of clients confusion matrix of the final model)**.
+ **Number of instructions executed per client during the training phases**. 
+ **Total execution time per client for the training phases**.
+ **Maximum RAM usage per client**.

#### Methods

##### Abstract Methods
- `get_skeleton_model(self) -> Model`: Returns the Keras model skeleton.
- `get_classes_names(self) -> list[str]`: Returns a list of class names used for generating the confusion matrix.

##### Public Methods
- `run(self) -> None`: Starts the server and manages the entire federated learning cycle.
- `enable_clients_profiling(self, value: bool) -> None`: Enables profiling to receive key performance indicators (KPIs) from the nodes.
- `enable_evaluations_plots(self, value: bool) -> None`: Displays evaluation and KPI plots if enabled.
- `set_aggregation_algorithm(self, aggregation_algorithm) -> None`: Sets the weight aggregation algorithm.
- `save_federated_weights(self, file_path) -> None`: Saves the federated model to the specified file path.
- `load_initial_weights(self, file_path) -> None`: Initializes the federated model with weights from the specified path.


### Client
The client component is defined by the abstract class [TCPClient](/src/TCPClient.py). The constructor requires two main parameters:
- `server_address`: The server address to connect to.
- `client_id`: The ID of the client.

Abstract class TCPClient  is implemented by the [Client](/src/Client.py) Class. The constructor of the Client Class requires also: 

- `Num_classes`: The number of output classes of the model (e.g. 10 for digits 0–9). Used to define the output layer. 
- `Batch_size`: The number of training samples used in one forward/backward pass during local training on each client. 
- `Train_epochs`: Number of epochs each client trains locally before sending weights back to the server for aggregation. 
- `model`: The path in which the model class is saved (e.g. src.models.model1). This class must implement `get_skeleton_model()`
- `Dataset_loader`: The path in which the dataset Loader class is saved (e.g. `src.dataloaders.bnci.datasetLoader`)
- `Loss`: The name of the loss function to be used for training (e.g. `"categorical_crossentropy"`, `"mse"`). Must match a valid Keras loss class. 
- `Metric`: The name of the evaluation metric used during training/validation (e.g. `"accuracy"`, `"mae"`). Must match a valid Keras metric class. 
- `Optimizer`: The name of the optimizer to be used (e.g. `"Adam"`, `"SGD"`). Must match a valid Keras optimizer class. 
- `Loss_params`: A JSON-formatted dictionary with parameters for the loss function. For example: `{"from_logits": false}`. 
- `Metric_params`: A JSON-formatted dictionary with parameters for the metric. For example: `{}` or `{"top_k": 5}`. 
- `Optimizer_params`: A JSON-formatted dictionary with parameters for the optimizer. For example: `{"learning_rate": 0.001}`. 
- `Input_shape`: The input shape of the model (e.g. 253,1). Must be formatted as a comma-separated string, and converted to a tuple at runtime. 



The ```run()``` is executed to run the client. 

The client opens a socket and connects to the server's address. Then, it waits to receive the federated model from the server. Once it receives the weights and biases, it loads them into the local model (net) and starts an initial evaluation. In this phase, the evaluation helps understanding the accuracy of the federated model using the local test dataset. After evaluating the federated model, the client starts the training on the training data.

The dataset is divided into batches, where each batch has a number of samples equal to the value returned by the ```get_batch_size()``` method. Training proceeds for a number of epochs equal to the value returned by the ```get_train_epochs()``` method. Upon completion, the model is re-evaluated on the test data, and the results are stored. Afterward, the client sends the weights and biases of the just-trained model to the server. This operation repeats until the server sends the final model, on which the client performs a single evaluation, sending all previous evaluations back to the server.

Once the federated training is complete, the client closes the connection with the server.

#### Methods

##### Abstract Methods
- `load_dataset(self) -> tuple`: Loads the client's dataset, returning training and test sets.
- `get_skeleton_model(self) -> keras.Model`: Returns the model to be trained.
- `get_optimizer(self) -> keras.optimizers.Optimizer | str`: Returns the optimizer for compiling the model.
- `get_loss_function(self) -> keras.losses.Loss | str`: Returns the loss function for compiling the model.
- `get_metric(self) -> keras.metrics.Metric | str`: Returns the metric for evaluating the model.
- `get_batch_size(self) -> int`: Returns the batch size for training.
- `get_train_epochs(self) -> int`: Returns the number of training epochs.
- `get_num_classes(self) -> int`: Returns the number of classes in the client's dataset.

##### Public Methods
- `run(self) -> None`: Executes the main operations of the client.
- `enable_op_determinism(self) -> None`: Configures training to use deterministic operations, ensuring reproducibility of experimental results.
- `shuffle_dataset_each_epoch(self, value: bool) -> None`: Enables or disables shuffling the training dataset at the beginning of each epoch. Enabled by default.


### Supported aggregation algorithms
The framework supports several aggregation algorithms:

- **FedAvg**: Aggregates the weights of the clients' models by computing a weighted or simple average.
  - Weighted Average: Each local model contributes based on the number of training samples used by the client.
  - Simple Average: Each local model contributes equally to the computation of the federated model.
  
  Weighted Average Formula:
 ``````
  Simple Average Formula:
    ![equation](https://latex.codecogs.com/svg.image?\theta_{t&plus;1}=\frac{\sum_{k=1}^{K}n_k\theta_t^{(k)}}{n})


- **FedMiddleAvg**: Averages the current federated model with the average of the clients' models calculated using FedAvg.
  \[
  \theta_{t+1} = \frac{(\theta_{t+1}) + \theta_t}{2}
  \]

- **FedAvgMomentum**: Integrates the concept of momentum into the aggregation of weights to accelerate convergence and reduce oscillations.
  \[
  (\theta_{t+1}) = FedAvg(\theta_t^{(1)}, \theta_t^{(2)}, ..., \theta_t^{(k)})
  \]
  \[
  \Delta_{t+1} = (\theta_{t+1}) - \theta_t
  \]
  \[
  v_{t+1} = \beta v_t + \Delta_{t+1} \quad \beta \in [0, 1]
  \]
  \[
  \theta_{t+1} = \theta_t + \eta v_{t+1}
  \]

- **FedAdam**: A variant of the Adam algorithm designed for federated learning, combining the benefits of AdaGrad and RMSProp.
  \[
  (\theta_{t+1}) = FedAvg(\theta_t^{(1)}, \theta_t^{(2)}, ..., \theta_t^{(k)})
  \]
  \[
  \Delta_{t+1} = (\theta_{t+1}) - \theta_t
  \]
  \[
  m_{t+1} = \beta_1 m_t + (1 - \beta_1) \Delta_{t+1}
  \]
  \[
  v_{t+1} = \beta_2 v_t + (1 - \beta_2) \Delta_{t+1}^2
  \]
  \[
  \theta_{t+1} = \theta_t + \eta \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
  \]

- **FedSGD**: A direct extension of Stochastic Gradient Descent (SGD) for federated learning.
  \[
  \theta_{t+1}^{(k)} = \theta_t - \eta g_t^{(k)}
  \]
  \[
  \theta_{t+1} = \frac{\sum_{k=1}^{K} n_k \theta_{t+1}^{(k)}}{n}
  \]


### Creating your own Model 
Every model must be implemented as a single Python class inside a module (file), and must provide the following method: 
- `get_skeleton_model(input_shape) -> keras.Model`: This method builds and returns a Keras model given the input shape. 
You can find models as example in `/src/models/`

### Creating your own DatasetLoader 
Each dataset loader must be implemented as a single Python class inside a module (file), and must provide at least the following method: 
- `load_dataset(clientid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: This method must return the train/test split for the given client ID: (x_train, x_test, y_train, y_test) 
You can find dataset loaders as example in `/src/datasetLoaders/` 





### Message exchange
The server and clients communicate using the TCP/IP protocol. Each exchanged message is composed of a byte sequence, with the first four bytes indicating the message length. The message consists of the following fields:
- `type`: Message type.
- `body`: Message body.

Defined message types include:
- `FEDERATED_WEIGHTS`: Contains the federated model created by the server.
- `CLIENT_MODEL`: Contains the model trained by a client.
- `END_FL_TRAINING`: Indicates the end of federated learning.
- `CLIENT_EVALUATION`: Sends client evaluations for each round.

The message is serialized using the *pickle module*, which transforms the message into a sequence of bytes. The generated sequence is concatenated with the initial 4 bytes representing the total length of the message.

### Profiling
Through the initial configurations, the server can decide whether to enable profiling on the nodes or not. If enabled, each node will, at the end of federated learning, send the following information to the server:

+ ```training_n_instructions```: number of instructions executed during the training phases.
+ ```training_execution_time```: total execution time of the training phases.
+ ```max_ram_used```: maximum RAM used by the node.
+ ```bytes_input```: number of bytes downloaded (received) by the client.
+ ```train_samples```: number of samples involved in the training of the local model.
+ ```test_samples```: number of samples involved in the test evaluation of the local/federated model.

The server will also save:

+ ```bytes_output```: number of bytes uploaded (sent) to each client.




## Distributed Deployment With Kubernetes and Helm
In this section we will provide a brief description of how configuration files works and then we will show how to setup a distributed environment running the software using Kubernetes and Helm. To offer a ready-to-go solution we will use Minikube to test the approach but you can setup your distributed Kubernetes Cluster. Launch the deployment scripts in a real distributed architecture needs just to substitute Minikube Cluster with a real Kubernetes Cluster. 

### Helm Configuration 
The values.yaml file defines the configuration of a Federated Learning experiment and can be customized before deploying the system with Helm. Users should modify the following fields to match their experiment setup: 
    -   **replicas.client**: Set the number of clients participating in the experiment. 
    -   **experiment**: Choose a name for the experiment. This name will also be used to store evaluation results. 
    -   **server section**: Define the server image, communication port, number of training rounds, input shape of the model, and output class labels. 
    -   **client section**: Specify the client Docker image and configuration: 
        -   **model**: Path to the Python class implementing the neural network model. 
        -   **datasetLoader**: Path to the class that loads and partitions the dataset. 
        -   **loss, metric, optimizer**: Names of Keras components used during training. 
        -   **lossParams, metricParams, optimizerParams**: JSON strings with parameters for each component. 
        -   **numClasses**: Number of output classes. 
        -   **batchSize, trainEpochs**: Training hyperparameters for each client. 
    -   **volumes section**: Defines the persistent volumes used to store logs and experiment results. These should generally not be modified unless you need to change the storage path or size. 



### Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
docker version
```

### Install Minikube

```bash
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
```

### Start Minikube and Check Minikube Installation
> [!IMPORTANT]
> Substitute $HOSTPATH with your path! 

```bash
minikube start --mount –mount-string="$HOSTPATH:/output" 
kubectl get nodes
```

### Install Helm
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

### Building Docker Images 

> [!IMPORTANT]
> Before building images you have to switch to  Minikube Env!
```bash
eval $(minikube docker-env)
docker build -t fed-server:latest -f Dockerfile.server .
docker build -t fed-client:latest -f Dockerfile.client .
```

### Configure Execution
To setup your experiment just modify Helm configuration files. Specifically, you have to modify
```bash
helm/values.yaml
```
in which you can setup the experiment folder, the server port you want to use and the number of clients to deploy.


###  Deploy with Helm
Inside the `helm/` folder:
```bash
helm install fl-demo .
kubectl get pods
```
To scale:
```bash
helm upgrade fl-demo . --set replicas.client=5
```
To see Logs: 
```
Kubectl logs pod_name 
```

### Cleanup
Inside the `helm/` folder:

```bash
helm uninstall fl-demo 
minikube stop
minikube delete
```

### Limits of the implementation
The proposed implementation is very simple, and for this reason, some simplifications were necessary, leading to the following limitations:

+ Server and client remain connected until the end of federated learning. In a realistic scenario, clients might participate only in certain rounds, or, better yet, clients may not be available simultaneously.
+ Dynamic addition of new clients during the learning phase is not supported. The clients participating in learning will remain connected until the last round. No one can opt-out or join later.
+ Security aspects are missing; anyone can connect to the server.
+ The exchanged messages are not encrypted.
+ ...and so on.

