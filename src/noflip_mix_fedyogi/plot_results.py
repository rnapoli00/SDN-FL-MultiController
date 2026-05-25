import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import os
import json

CLASS_NAMES = ["benign", "ack", "syn", "fin", "udp"]
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]   # un colore per client
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # un colore per classe


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def load_client_files():
    """Carica tutti i CSV dei client e restituisce un dict {client_id: df}."""
    files = sorted(glob.glob("metrics_client_*.csv"))
    if not files:
        raise FileNotFoundError("Nessun file metrics_client_*.csv trovato. Avvia prima l'addestramento!")
    clients = {}
    for f in files:
        cid = f.split("_")[-1].replace(".csv", "")
        clients[cid] = pd.read_csv(f)
    return clients


def split_sessions(df):
    """
    Divide un DataFrame in sessioni distinte.
    Una nuova sessione inizia ogni volta che il valore di 'round' torna a 1.
    """
    sessions = []
    restart_indices = df.index[df["round"] == 1].tolist()

    if not restart_indices:
        return [df.reset_index(drop=True)]

    boundaries = restart_indices + [len(df)]

    for start, end in zip(boundaries, boundaries[1:]):
        session_df = df.iloc[start:end].reset_index(drop=True)
        if not session_df.empty:
            sessions.append(session_df)

    return sessions


def split_all_clients_by_session(clients):
    """
    Dato {client_id: df}, restituisce una lista di dict {client_id: df_sessione}.
    """
    per_client_sessions = {cid: split_sessions(df) for cid, df in clients.items()}
    n_sessions = min(len(s) for s in per_client_sessions.values())
    
    if n_sessions == 0:
        return [clients]

    session_list = []
    for s_idx in range(n_sessions):
        session_clients = {cid: per_client_sessions[cid][s_idx] for cid in clients}
        session_list.append(session_clients)

    return session_list


def split_server_by_session(server_df):
    """Divide il dataframe del server nelle stesse sessioni."""
    if server_df is None or server_df.empty:
        return [server_df]
    return split_sessions(server_df)


def has_extended_metrics(df):
    """Restituisce True se il CSV contiene le colonne recall/f1/cm_r_c."""
    return f"recall_{CLASS_NAMES[0]}" in df.columns and "cm_0_0" in df.columns


def extract_confmat(row):
    """Ricostruisce la matrice 5x5 dalle colonne cm_r_c."""
    return np.array([[row[f"cm_{r}_{c}"] for c in range(5)] for r in range(5)], dtype=int)


# ──────────────────────────────────────────────────────────────
# Pagina 1 – Loss & Accuracy
# ──────────────────────────────────────────────────────────────

def plot_loss_accuracy(clients, server_df, session_label=""):
    title_suffix = f" — {session_label}" if session_label else ""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Loss & Accuracy{title_suffix}", fontsize=14, fontweight="bold")

    # ── Client loss ──
    ax = axes[0, 0]
    for i, (cid, df) in enumerate(clients.items()):
        ax.plot(df["round"], df["loss"], marker="o", color=COLORS[i], label=f"Client {cid}")
    ax.set_title("Loss per client")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Round federato")
    ax.set_ylabel("Cross Entropy Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # ── Client accuracy ──
    ax = axes[0, 1]
    for i, (cid, df) in enumerate(clients.items()):
        ax.plot(df["round"], df["accuracy"], marker="o", color=COLORS[i], label=f"Client {cid}")
    ax.set_title("Accuracy per client")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Round federato")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # ── Server distributed loss ──
    ax = axes[1, 0]
    if server_df is not None and not server_df.empty and "distributed_loss" in server_df.columns:
        ax.plot(server_df["round"], server_df["distributed_loss"],
                marker="s", color="#E91E63", linewidth=2, label="Server distributed loss")
        ax.set_title("Distributed Loss (server)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Round federato")
        ax.set_ylabel("Loss aggregata")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "server_metrics.csv\nnon trovato", ha="center", va="center", color="gray")
        ax.set_title("Distributed Loss (server)")

    # ── Server distributed accuracy ──
    ax = axes[1, 1]
    if server_df is not None and not server_df.empty and "accuracy" in server_df.columns:
        ax.plot(server_df["round"], server_df["accuracy"],
                marker="D", color="#7B1FA2", linewidth=2, label="Server accuracy (pesata)")
        ax.set_title("Distributed Accuracy (server)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Round federato")
        ax.set_ylabel("Accuracy aggregata (media pesata)")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=server_df["accuracy"].max(), color="#CE93D8", linestyle=":", linewidth=1.2,
                   label=f"Best: {server_df['accuracy'].max():.4f}")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "Colonna 'accuracy' non trovata", ha="center", va="center", color="gray")
        ax.set_title("Distributed Accuracy (server)")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Pagine 2, 3, 4 – Metriche per classe
# ──────────────────────────────────────────────────────────────

def plot_metric_per_class(clients, metric_name, session_label=""):
    title_suffix = f" — {session_label}" if session_label else ""
    n_clients = len(clients)
    fig, axes = plt.subplots(1, n_clients, figsize=(6 * n_clients, 5), sharey=True)
    if n_clients == 1:
        axes = [axes]
    
    titles = {"recall": "Recall", "f1": "F1-score", "precision": "Precision"}
    fig.suptitle(f"{titles.get(metric_name, metric_name)} per classe e per round{title_suffix}", fontsize=14, fontweight="bold")

    markers = {"recall": "o", "f1": "s", "precision": "^"}
    
    for ax, (cid, df) in zip(axes, clients.items()):
        for j, cls in enumerate(CLASS_NAMES):
            col = f"{metric_name}_{cls}"
            if col in df.columns:
                ax.plot(df["round"], df[col], marker=markers.get(metric_name, "o"), color=CLASS_COLORS[j], label=cls)
        ax.set_title(f"Client {cid}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Round federato")
        ax.set_ylabel(titles.get(metric_name, metric_name))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Pagina 5 – Confusion matrix (ultimo round disponibile)
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrices(clients, session_label=""):
    title_suffix = f" — {session_label}" if session_label else ""
    n_clients = len(clients)
    fig, axes = plt.subplots(1, n_clients, figsize=(5 * n_clients, 4))
    if n_clients == 1:
        axes = [axes]
    fig.suptitle(f"Confusion Matrix — ultimo round{title_suffix}", fontsize=14, fontweight="bold")

    for ax, (cid, df) in zip(axes, clients.items()):
        if "cm_0_0" not in df.columns:
            ax.text(0.5, 0.5, "Dati non disponibili", ha="center", va="center", color="gray")
            ax.set_title(f"Client {cid}")
            continue

        last_row = df.iloc[-1]
        cm = extract_confmat(last_row)

        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Client {cid} (round {int(last_row['round'])})")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predetto")
        ax.set_ylabel("Vero")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for r in range(5):
            for c in range(5):
                color = "white" if cm_norm[r, c] > 0.6 else "black"
                ax.text(c, r, str(int(cm[r, c])), ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Pagina 6 – Distribuzione dei dati (k=5 categorie)
# ──────────────────────────────────────────────────────────────

def plot_k5_distribution(clients, session_label=""):
    """
    Crea un grafico a barre in pila (100% stacked bar chart) per mostrare 
    la distribuzione esatta delle 5 classi (il sample di Dirichlet) per ogni client.
    """
    title_suffix = f" — {session_label}" if session_label else ""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    client_ids = []
    distributions = []
    
    for cid, df in clients.items():
        if "cm_0_0" not in df.columns:
            continue

        last_row = df.iloc[-1]
        cm = extract_confmat(last_row)
        
        # Le vere occorrenze delle classi si trovano sommando per riga la matrice
        true_counts = cm.sum(axis=1) 
        total = true_counts.sum()
        
        if total == 0:
            props = np.zeros(5)
        else:
            props = true_counts / total
            
        client_ids.append(f"Client {cid}")
        distributions.append(props)

    if not distributions:
        ax.text(0.5, 0.5, "Nessun dato disponibile", ha="center", va="center")
        return fig

    distributions = np.array(distributions) # shape (n_clients, 5)
    
    bottom = np.zeros(len(client_ids))
    
    for j, cls_name in enumerate(CLASS_NAMES):
        bars = ax.bar(client_ids, distributions[:, j], label=cls_name, color=CLASS_COLORS[j], bottom=bottom)
        
        # Aggiungi le percentuali al centro di ogni segmento (se il blocco è visibile, > 3%)
        for i, bar in enumerate(bars):
            val = distributions[i, j]
            if val > 0.03:
                ax.text(bar.get_x() + bar.get_width()/2, 
                        bottom[i] + val/2, 
                        f"{val*100:.1f}%", 
                        ha='center', va='center', color='white', fontsize=10, fontweight='bold')
                
        bottom += distributions[:, j]

    ax.set_ylabel("Proporzione nel Test Set")
    ax.set_title(f"Distribuzione delle Classi (k=5) per Client{title_suffix}", fontsize=14, fontweight="bold")
    ax.legend(title="Classi", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Salvataggio
# ──────────────────────────────────────────────────────────────

def save_session_plots(session_clients, session_server_df, suffix, extended):
    label = f"{suffix}"

    # 1. Loss & Accuracy
    fig1 = plot_loss_accuracy(session_clients, session_server_df, session_label=label)
    name1 = f"plot_loss_accuracy_{suffix}.png"
    fig1.savefig(name1, dpi=150)
    plt.close(fig1)
    print(f"Salvato: {name1}")

    if extended:
        # 2,3,4. Metriche
        for metric in ["recall", "f1", "precision"]:
            fig_m = plot_metric_per_class(session_clients, metric, session_label=label)
            name_m = f"plot_{metric}_per_class_{suffix}.png"
            fig_m.savefig(name_m, dpi=150)
            plt.close(fig_m)
            print(f"Salvato: {name_m}")

        # 5. Confusion Matrix
        fig5 = plot_confusion_matrices(session_clients, session_label=label)
        name5 = f"plot_confusion_matrices_{suffix}.png"
        fig5.savefig(name5, dpi=150)
        plt.close(fig5)
        print(f"Salvato: {name5}")
        
        # 6. Distribuzione k=5 (Barre in pila)
        fig6 = plot_k5_distribution(session_clients, session_label=label)
        name6 = f"plot_k5_distribution_{suffix}.png"
        fig6.savefig(name6, dpi=150)
        plt.close(fig6)
        print(f"Salvato: {name6}")
    else:
        print(f"  [{label}] Metriche estese mancanti: saltati plot per classe/cm/distribuzione.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def plot_federated_results():
    clients = load_client_files()

    server_df = None
    if os.path.isfile("server_metrics.csv"):
        server_df = pd.read_csv("server_metrics.csv")
    else:
        print("Attenzione: server_metrics.csv non trovato.")

    session_clients_list = split_all_clients_by_session(clients)
    session_server_list  = split_server_by_session(server_df)

    n_sessions = len(session_clients_list)
    while len(session_server_list) < n_sessions:
        session_server_list.append(None)

    print(f"\nSessioni rilevate: {n_sessions}\n")

    for s_idx, (session_clients, session_server) in enumerate(
            zip(session_clients_list, session_server_list), start=1):

        suffix   = f"session{s_idx}"
        extended = any(has_extended_metrics(df) for df in session_clients.values())

        print(f"── Sessione {s_idx} ──────────────────────────────────────")
        save_session_plots(session_clients, session_server, suffix, extended)

    print("\nGrafici generati e salvati correttamente.")


if __name__ == "__main__":
    plot_federated_results()