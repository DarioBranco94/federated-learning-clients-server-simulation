#!/bin/bash

echo "Avviando Minikube..."
minikube start

# Configura l'ambiente Docker per Minikube
echo "Configurando Docker per Minikube..."
eval $(minikube -p minikube docker-env)

# Monta la prima cartella in background
echo "Montando la cartella logs..."
nohup minikube mount "/home/dario/Documents/federated-learning-clients-server-simulation/Outputs/logs:/logs" > /dev/null 2>&1 &

# Monta la seconda cartella in background
echo "Montando la cartella evaluations..."
nohup minikube mount "/home/dario/Documents/federated-learning-clients-server-simulation/Outputs/evaluations:/evaluations" > /dev/null 2>&1 &

# Attendere qualche secondo per assicurarsi che i mount siano attivi
sleep 3

# Costruzione delle immagini Docker
echo "Costruendo l'immagine Docker del client..."
docker build -t fed-client:latest -f Dockerfile.client .

echo "Costruendo l'immagine Docker del server..."
docker build -t fed-client:server -f Dockerfile.server .

echo "Minikube avviato, Docker configurato, cartelle montate e immagini buildate con successo!"
