import flwr as fl

def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # tutti i client partecipano a ogni round
        min_fit_clients=1,
        min_available_clients=1,
    )

    fl.server.start_server(server_address="192.168.56.105:3333", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

if __name__ == "__main__":
    start_server()
