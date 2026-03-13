import sys
import flwr as fl

def start_server():

    # Legge il numero di client dagli argomenti (default a 2 se non specificato)
    n_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Configurazione server per {n_clients} controller.")
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # tutti i client partecipano a ogni round
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        min_evaluate_clients=n_clients,
    )

    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)

if __name__ == "__main__":
    start_server()

