# run_minikube_windows.ps1

Write-Output "Avviando Minikube..."
minikube start

# Configura l'ambiente Docker per Minikube
Write-Output "Configurando Docker per Minikube..."
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

# Monta la prima cartella in background
Write-Output "Montando la cartella data..."
Start-Process powershell -ArgumentList "-NoExit", "-Command minikube mount 'C:\Users\BRNDRA94B21B715D\Documents\test\federated-learning-clients-server-simulation\data\outputs:/output'" -WindowStyle Hidden

# Costruzione delle immagini Docker
Write-Output "Costruendo l'immagine Docker del client..."
docker build -t fed-client:latest -f Dockerfile.client .

Write-Output "Costruendo l'immagine Docker del server..."
docker build -t fed-server:latest -f Dockerfile.server .

Write-Output "Minikube avviato, Docker configurato, cartelle montate e immagini buildate con successo!"
