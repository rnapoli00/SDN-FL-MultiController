import sys
import csv
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union

import flwr as fl
from flwr.common import Metrics


# ─────────────────────────────────────────────────────────────────────────────
# Funzione di aggregazione metriche di evaluation (chiamata dal server
# dopo che tutti i client hanno risposto al round di evaluate).
#
# Riceve: List[ (num_examples, {"accuracy": float, ...}) ]
# Ritorna: Dict aggregato  {"accuracy": float_pesata}
# ─────────────────────────────────────────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Calcola l'accuracy media pesata sul numero di campioni di ogni client.
    Aggrega anche precision, recall e f1 (macro) se i client le inviano.
    """
    if not metrics:
        return {}

    total_examples = sum(n for n, _ in metrics)

    aggregated: Dict[str, float] = {}
    # Raccogli tutte le chiavi metriche disponibili
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())

    for key in all_keys:
        weighted_sum = sum(
            n * m[key] for n, m in metrics if key in m
        )
        aggregated[key] = weighted_sum / total_examples

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Server principale
# ─────────────────────────────────────────────────────────────────────────────
def start_server():
    n_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Configurazione server per {n_clients} controller.")

    def on_evaluate_config_fn(server_round: int):
        return {"server_round": server_round}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                       # tutti i client al fit
        fraction_evaluate=1.0,                  # tutti i client all'evaluate
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        min_evaluate_clients=n_clients,
        on_evaluate_config_fn=on_evaluate_config_fn,
        # ── NUOVO: aggrega le metriche restituite da client.evaluate() ──
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Salvataggio metriche per round
    #
    # history.losses_distributed      → [(round, loss), ...]
    # history.metrics_distributed     → {"accuracy": [(round, val), ...], ...}
    # ─────────────────────────────────────────────────────────────────────

    # Costruiamo un dizionario {round: {metric: value}}
    per_round: Dict[int, Dict] = {}

    # --- loss ---
    for server_round, loss in history.losses_distributed:
        per_round.setdefault(server_round, {})["distributed_loss"] = round(loss, 6)

    # --- metriche aggregate (accuracy + eventuali altre) ---
    for metric_name, round_values in history.metrics_distributed.items():
        for server_round, value in round_values:
            per_round.setdefault(server_round, {})[metric_name] = round(float(value), 6)

    # Stampa riepilogo a video
    print("\n" + "=" * 60)
    print("  RIEPILOGO METRICHE SERVER PER ROUND")
    print("=" * 60)
    header_keys = sorted(
        {k for d in per_round.values() for k in d if k != "distributed_loss"}
    )
    col_header = f"  {'Round':>5}  {'Loss':>10}  " + "  ".join(f"{k:>10}" for k in header_keys)
    print(col_header)
    print("-" * 60)
    for rnd in sorted(per_round):
        d = per_round[rnd]
        loss_str = f"{d.get('distributed_loss', float('nan')):>10.6f}"
        metric_str = "  ".join(
            f"{d.get(k, float('nan')):>10.6f}" for k in header_keys
        )
        print(f"  {rnd:>5}  {loss_str}  {metric_str}")
    print("=" * 60)

    # --- scrittura CSV ---
    log_path = "server_metrics.csv"
    all_metric_keys = ["distributed_loss"] + header_keys

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "timestamp"] + all_metric_keys)
        for rnd in sorted(per_round):
            d = per_round[rnd]
            row = [
                rnd,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ] + [round(d.get(k, float("nan")), 6) for k in all_metric_keys]
            writer.writerow(row)

    print(f"\n[Server] Metriche per round salvate in: {log_path}")


if __name__ == "__main__":
    start_server()
