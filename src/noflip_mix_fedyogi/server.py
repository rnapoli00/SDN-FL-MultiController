import sys
import flwr as fl
from flwr.common import ndarrays_to_parameters
# Importa la classe dal tuo file client per coerenza dei pesi
from fl_client import NeuralNetwork, get_parameters 
import torch

def start_server():
    '''
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # tutti i client partecipano a ogni round
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
    )
    '''
    # Legge il numero di client dagli argomenti (default a 2 se non specificato)
    n_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Configurazione server per {n_clients} controller.")
    
    # Inizializzazione parametri (Obbligatoria per FedYogi)
    net = NeuralNetwork()
    initial_params = ndarrays_to_parameters(get_parameters(net))

    strategy = fl.server.strategy.FedYogi(
        fraction_fit=1.0,
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        min_evaluate_clients=n_clients,
        initial_parameters=initial_params,
        eta=0.01,  # η (eta)
        beta_1=0.9,
        beta_2=0.999,
        tau=1e-3,                   # Grado di adattatività
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080", 
        config=fl.server.ServerConfig(num_rounds=5), 
        strategy=strategy
    )


if __name__ == "__main__":
    start_server()

