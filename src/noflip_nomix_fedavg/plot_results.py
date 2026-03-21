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
    Una nuova sessione inizia ogni volta che il valore di 'round' torna a 1
    (dopo non essere già all'inizio del file).
    Restituisce una lista di DataFrame, uno per sessione.
    """
    sessions = []
    # Trova gli indici di riga dove round == 1
    restart_indices = df.index[df["round"] == 1].tolist()

    if not restart_indices:
        # Nessun round==1 trovato: restituisci l'intero df come unica sessione
        return [df.reset_index(drop=True)]

    # Aggiungi la fine del dataframe come sentinella
    boundaries = restart_indices + [len(df)]

    for start, end in zip(boundaries, boundaries[1:]):
        session_df = df.iloc[start:end].reset_index(drop=True)
        if not session_df.empty:
            sessions.append(session_df)

    return sessions


def split_all_clients_by_session(clients):
    """
    Dato {client_id: df}, restituisce una lista di dict {client_id: df_sessione},
    uno per ogni sessione. Tutte i client devono avere lo stesso numero di sessioni;
    se non coincidono si usa il minimo comune.
    """
    # Calcola le sessioni per ogni client
    per_client_sessions = {cid: split_sessions(df) for cid, df in clients.items()}

    n_sessions = min(len(s) for s in per_client_sessions.values())
    if n_sessions == 0:
        return [clients]  # fallback: tratta tutto come sessione unica

    session_list = []
    for s_idx in range(n_sessions):
        session_clients = {cid: per_client_sessions[cid][s_idx] for cid in clients}
        session_list.append(session_clients)

    return session_list


def split_server_by_session(server_df):
    """Divide il dataframe del server nelle stesse sessioni (round che ricomincia da 1)."""
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Loss & Accuracy{title_suffix}", fontsize=14, fontweight="bold")

    # Client loss
    ax = axes[0]
    for i, (cid, df) in enumerate(clients.items()):
        ax.plot(df["round"], df["loss"], marker="o", color=COLORS[i], label=f"Client {cid}")
    ax.set_title("Loss per singolo client")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Round federato")
    ax.set_ylabel("Cross Entropy Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # Client accuracy
    ax = axes[1]
    for i, (cid, df) in enumerate(clients.items()):
        ax.plot(df["round"], df["accuracy"], marker="o", color=COLORS[i], label=f"Client {cid}")
    ax.set_title("Accuracy per client")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("Round federato")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # Server distributed loss
    ax = axes[2]
    if server_df is not None and not server_df.empty:
        ax.plot(server_df["round"], server_df["distributed_loss"],
                marker="s", color="#E91E63", linewidth=2, label="Server distributed loss")
        ax.set_title("Loss Distribuita (server)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Round federato")
        ax.set_ylabel("Loss aggregata")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "server_metrics.csv\nnon trovato", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Distributed Loss (server)")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Pagina 2 – Recall per classe
# ──────────────────────────────────────────────────────────────

def plot_recall_per_class(clients, session_label=""):
    title_suffix = f" — {session_label}" if session_label else ""
    n_clients = len(clients)
    fig, axes = plt.subplots(1, n_clients, figsize=(6 * n_clients, 5), sharey=True)
    if n_clients == 1:
        axes = [axes]
    fig.suptitle(f"Recall per classe e per round{title_suffix}", fontsize=14, fontweight="bold")

    for ax, (cid, df) in zip(axes, clients.items()):
        for j, cls in enumerate(CLASS_NAMES):
            col = f"recall_{cls}"
            if col in df.columns:
                ax.plot(df["round"], df[col], marker="o", color=CLASS_COLORS[j], label=cls)
        ax.set_title(f"Client {cid}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Round federato")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Pagina 3 – F1 per classe
# ──────────────────────────────────────────────────────────────

def plot_f1_per_class(clients, session_label=""):
    title_suffix = f" — {session_label}" if session_label else ""
    n_clients = len(clients)
    fig, axes = plt.subplots(1, n_clients, figsize=(6 * n_clients, 5), sharey=True)
    if n_clients == 1:
        axes = [axes]
    fig.suptitle(f"F1-score per classe e per round{title_suffix}", fontsize=14, fontweight="bold")

    for ax, (cid, df) in zip(axes, clients.items()):
        for j, cls in enumerate(CLASS_NAMES):
            col = f"f1_{cls}"
            if col in df.columns:
                ax.plot(df["round"], df[col], marker="s", color=CLASS_COLORS[j], label=cls)
        ax.set_title(f"Client {cid}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Round federato")
        ax.set_ylabel("F1-score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Pagina 4 – Precision per classe
# ──────────────────────────────────────────────────────────────

def plot_precision_per_class(clients, session_label=""):
    title_suffix = f" — {session_label}" if session_label else ""
    n_clients = len(clients)
    fig, axes = plt.subplots(1, n_clients, figsize=(6 * n_clients, 5), sharey=True)
    if n_clients == 1:
        axes = [axes]
    fig.suptitle(f"Precision per classe e per round{title_suffix}", fontsize=14, fontweight="bold")

    for ax, (cid, df) in zip(axes, clients.items()):
        for j, cls in enumerate(CLASS_NAMES):
            col = f"precision_{cls}"
            if col in df.columns:
                ax.plot(df["round"], df[col], marker="^", color=CLASS_COLORS[j], label=cls)
        ax.set_title(f"Client {cid}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlabel("Round federato")
        ax.set_ylabel("Precision")
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
    fig.suptitle(f"Confusion Matrix — Ultimo round{title_suffix}", fontsize=14, fontweight="bold")

    for ax, (cid, df) in zip(axes, clients.items()):
        if "cm_0_0" not in df.columns:
            ax.text(0.5, 0.5, "Dati non disponibili\n(formato CSV vecchio)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"Client {cid}")
            continue

        last_row = df.iloc[-1]
        cm = extract_confmat(last_row)

        # Normalizza per riga (recall visiva)
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Client {cid}")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predetto")
        ax.set_ylabel("Vero")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotazioni con i conteggi assoluti
        for r in range(5):
            for c in range(5):
                color = "white" if cm_norm[r, c] > 0.6 else "black"
                ax.text(c, r, str(int(cm[r, c])), ha="center", va="center",
                        fontsize=7, color=color)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Salvataggio di una singola sessione
# ──────────────────────────────────────────────────────────────

def save_session_plots(session_clients, session_server_df, suffix, extended):
    """Genera e salva tutti i grafici per una singola sessione."""
    label = f"{suffix}"

    fig1 = plot_loss_accuracy(session_clients, session_server_df, session_label=label)
    name1 = f"plot_loss_accuracy_{suffix}.png"
    fig1.savefig(name1, dpi=150)
    plt.close(fig1)
    print(f"Salvato: {name1}")

    if extended:
        fig2 = plot_recall_per_class(session_clients, session_label=label)
        name2 = f"plot_recall_per_class_{suffix}.png"
        fig2.savefig(name2, dpi=150)
        plt.close(fig2)
        print(f"Salvato: {name2}")

        fig3 = plot_f1_per_class(session_clients, session_label=label)
        name3 = f"plot_f1_per_class_{suffix}.png"
        fig3.savefig(name3, dpi=150)
        plt.close(fig3)
        print(f"Salvato: {name3}")

        fig4 = plot_precision_per_class(session_clients, session_label=label)
        name4 = f"plot_precision_per_class_{suffix}.png"
        fig4.savefig(name4, dpi=150)
        plt.close(fig4)
        print(f"Salvato: {name4}")

        fig5 = plot_confusion_matrices(session_clients, session_label=label)
        name5 = f"plot_confusion_matrices_{suffix}.png"
        fig5.savefig(name5, dpi=150)
        plt.close(fig5)
        print(f"Salvato: {name5}")
    else:
        print(f"  [{label}] Metriche estese non presenti: recall/f1/confusion_matrix saltati.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def plot_federated_results():
    clients = load_client_files()

    # Server metrics (opzionale)
    server_df = None
    if os.path.isfile("server_metrics.csv"):
        server_df = pd.read_csv("server_metrics.csv")
    else:
        print("Attenzione: server_metrics.csv non trovato. Il grafico server sarà vuoto.")

    # ── Suddivisione per sessioni ──────────────────────────────
    # Ogni volta che round torna a 1 inizia una nuova sessione.
    # Se c'è una sola sessione il suffisso sarà semplicemente "session1".
    session_clients_list = split_all_clients_by_session(clients)
    session_server_list  = split_server_by_session(server_df)

    n_sessions = len(session_clients_list)

    # Allinea le sessioni server a quelle dei client (padding con None se mancano)
    while len(session_server_list) < n_sessions:
        session_server_list.append(None)

    print(f"\nSessioni rilevate: {n_sessions}\n")

    for s_idx, (session_clients, session_server) in enumerate(
            zip(session_clients_list, session_server_list), start=1):

        suffix   = f"session{s_idx}"
        extended = any(has_extended_metrics(df) for df in session_clients.values())

        print(f"── Sessione {s_idx} ──────────────────────────────────────")
        save_session_plots(session_clients, session_server, suffix, extended)

    print("\nStatistiche sulle sessioni di addestramento salvate.")
    plt.show()


if __name__ == "__main__":
    plot_federated_results()