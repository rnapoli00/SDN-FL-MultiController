
import flwr as fl

def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # tutti i client partecipano a ogni round
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=2,
    )

    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

if __name__ == "__main__":
    start_server()

