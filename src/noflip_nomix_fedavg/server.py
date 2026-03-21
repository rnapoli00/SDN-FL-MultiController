import sys
import csv
import os
from datetime import datetime

import flwr as fl

def start_server():
    n_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Configurazione server per {n_clients} controller.")

    def on_evaluate_config_fn(server_round: int):
        return {"server_round": server_round}

    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # tutti i client partecipano a ogni round
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        min_evaluate_clients=n_clients,
        on_evaluate_config_fn=on_evaluate_config_fn
        
    )

    # start_server restituisce un oggetto History con tutte le metriche aggregate
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

    # history.losses_distributed è una lista di tuple (round, loss)
    log_path = "server_metrics.csv"
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["round", "timestamp", "distributed_loss"])
        for server_round, loss in history.losses_distributed:
            writer.writerow([
                server_round,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(loss, 6)
            ])

    print(f"[Server] Loss distribuita salvata in {log_path}")


if __name__ == "__main__":
    start_server()