# run_minikube_windows.ps1

$hostPath = "C:\Users\dario\Documents\federated-learning-clients-server-simulation\data\output"
$containerPath = "/output"

Write-Output "Avviando Minikube con il volume montato..."
minikube start --mount --mount-string="${hostPath}:${containerPath}"

Write-Output "Configurando Docker per Minikube..."
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

Write-Output "Costruendo l'immagine Docker del client..."
docker build -t fed-client:latest -f Dockerfile.client .

Write-Output "Costruendo l'immagine Docker del server..."
docker build -t fed-server:latest -f Dockerfile.server .

Write-Output "✅ Minikube avviato, Docker configurato, volume montato e immagini buildate con successo!"
