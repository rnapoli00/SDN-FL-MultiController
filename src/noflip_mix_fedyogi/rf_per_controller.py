"""
rf_per_controller.py
────────────────────
Modulo condiviso: analisi feature importance Random Forest per singolo controller.
Importato sia da feature_importance_rf.py che da feature_importance_rf_combined.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

FEATURE_COLS = [
    "delta_time", "eth_type", "ip_proto", "pkt_len",
    "tcp_fin", "tcp_syn", "tcp_rst", "tcp_psh", "tcp_ack", "tcp_urg"
]
TARGET_COL  = "target"
ALL_CLASSES = {0: "benign", 1: "ack", 2: "syn", 3: "fin", 4: "udp"}

# Colore fisso per ogni feature (coerente tra tutti i grafici)
FEATURE_COLORS = {
    "delta_time": "#64748B",
    "eth_type":   "#94A3B8",
    "ip_proto":   "#7C3AED",
    "pkt_len":    "#0891B2",
    "tcp_fin":    "#16A34A",
    "tcp_syn":    "#2563EB",
    "tcp_rst":    "#EA580C",
    "tcp_psh":    "#D97706",
    "tcp_ack":    "#DC2626",
    "tcp_urg":    "#DB2777",
}

CTRL_COLORS = ["#2563EB", "#16A34A", "#EA580C"]   # blu, verde, arancio


def run_rf_on_client(df, client_id):
    """
    Addestra una Random Forest sul dataset di un singolo client.
    Restituisce un dict con importances, accuracy, f1, confusion matrix e classi presenti.
    """
    present_classes = sorted(df[TARGET_COL].unique())
    class_names     = [ALL_CLASSES[c] for c in present_classes]

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Stratified split — gestisce anche classi con pochissimi campioni
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train_sc, y_train)

    y_pred = rf.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)

    # F1 macro su tutte le classi presenti
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_per   = f1_score(y_test, y_pred, average=None,
                        labels=present_classes, zero_division=0)

    mdi_imp = rf.feature_importances_
    mdi_std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)

    cm = confusion_matrix(y_test, y_pred, labels=present_classes, normalize='true')

    return {
        "client_id":  client_id,
        "mdi_imp":        mdi_imp,
        "mdi_std":        mdi_std,
        "acc":            acc,
        "f1_macro":       f1_macro,
        "f1_per":         f1_per,
        "cm":             cm,
        "present_classes": present_classes,
        "class_names":    class_names,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "dist":           pd.Series(y).value_counts().sort_index(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA A  — Feature importance orizzontale per ogni client (3 subplot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_client_importance(results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
    fig.patch.set_facecolor("#F8FAFC")

    bgs = ["#EFF6FF", "#F0FDF4", "#FFF7ED"]

    for ax, res, bg in zip(axes, results, bgs):
        cid   = res["client_id"]
        imp   = res["mdi_imp"]
        std   = res["mdi_std"]
        order = np.argsort(imp)[::-1]
        feats = np.array(FEATURE_COLS)

        colors = [FEATURE_COLORS[f] for f in feats[order]]
        bars = ax.barh(feats[order], imp[order], xerr=std[order],
                       color=colors, ecolor="#9CA3AF", capsize=3, alpha=0.88)

        ax.set_facecolor(bg)
        ax.invert_yaxis()
        ax.set_xlabel("Importanza MDI", fontsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.45)

        # Etichetta percentuale
        for bar, val in zip(bars, imp[order]):
            if val > 0.005:
                ax.text(val + max(imp)*0.012,
                        bar.get_y() + bar.get_height()/2,
                        f"{val*100:.1f}%", va='center', fontsize=8.5)

        # Classi presenti + accuracy
        cls_str = " + ".join(res["class_names"])
        ax.set_title(
            f"client {cid}\n"
            f"Classi: {cls_str}\n"
            f"Acc={res['acc']:.4f}  F1-macro={res['f1_macro']:.4f}",
            fontsize=11, fontweight='bold'
        )

    fig.suptitle(
        "Feature Importance MDI — Analisi per Singolo client\n",
        fontsize=13, fontweight='bold', color="#1E293B", y=1.02
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Per-client importance salvata: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA B  — Confronto ranking tra client (heatmap + linechart)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ranking_comparison(results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#F8FAFC")

    # ── B1. Heatmap importanze ──
    ax = axes[0]
    data_matrix = np.array([res["mdi_imp"] for res in results])  # (3, 10)
    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=data_matrix.max())
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"client {r['client_id']}\n({'+'.join(r['class_names'])})"
                        for r in results], fontsize=10)
    plt.colorbar(im, ax=ax, label="Importanza MDI")
    # Annotazioni numeriche
    for i in range(3):
        for j in range(len(FEATURE_COLS)):
            val = data_matrix[i, j]
            color = "white" if val > data_matrix.max()*0.6 else "#1E293B"
            ax.text(j, i, f"{val*100:.1f}%", ha='center', va='center',
                    fontsize=8.5, color=color, fontweight='bold')
    ax.set_title("Heatmap Importanze per client",
                 fontsize=12, fontweight='bold')
    ax.set_facecolor("#F8FAFC")

    # ── B2. Line chart ranking ──
    ax2 = axes[1]
    ax2.set_facecolor("#F8FAFC")
    ax2.grid(axis='y', linestyle='--', alpha=0.45)

    # Ordina feature per importanza media tra client
    mean_imp  = data_matrix.mean(axis=0)
    feat_order = np.argsort(mean_imp)[::-1]
    feats_sorted = [FEATURE_COLS[i] for i in feat_order]
    x = range(len(FEATURE_COLS))

    for res, color, marker in zip(results, CTRL_COLORS, ['o', 's', '^']):
        imp_sorted = [res["mdi_imp"][FEATURE_COLS.index(f)] for f in feats_sorted]
        ax2.plot(x, [v*100 for v in imp_sorted],
                 color=color, marker=marker, linewidth=2, markersize=7,
                 label=f"Ctrl {res['client_id']} ({'+'.join(res['class_names'])})")
        # Punti con valore
        for xi, val in zip(x, imp_sorted):
            if val > 0.005:
                ax2.text(xi, val*100 + 0.5, f"{val*100:.1f}%",
                         ha='center', fontsize=7.5, color=color)

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(feats_sorted, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel("Importanza MDI (%)", fontsize=11)
    ax2.set_title("Divergenza Ranking tra client\n",
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')

    fig.suptitle(
        "Confronto Feature Importance tra client",
        fontsize=13, fontweight='bold', color="#1E293B"
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Ranking comparison salvata: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA C  — Confusion matrix per ogni client (1 riga, 3 subplot)
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_client_cm(results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#F8FAFC")

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["cm"],
            display_labels=res["class_names"]
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
        ax.set_title(
            f"client {res['client_id']}\n"
            f"Acc={res['acc']:.4f}  F1={res['f1_macro']:.4f}",
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel("Predetto", fontsize=10)
        ax.set_ylabel("Reale", fontsize=10)

    fig.suptitle(
        "Confusion Matrix per client (normalizzata per riga)",
        fontsize=13, fontweight='bold', color="#1E293B"
    )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Per-client confusion matrix salvata: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FUNZIONE PRINCIPALE — esegue tutto e ritorna i risultati
# ─────────────────────────────────────────────────────────────────────────────
def run_per_client_analysis(csv_paths, base_path):
    """
    csv_paths: dict {client_id: path_csv}
    base_path: cartella dove salvare le figure
    Ritorna: lista di result dict (uno per client)
    """
    print("\n" + "="*60)
    print("  ANALISI PER SINGOLO client")
    print("="*60)

    # Verifica preventiva: tutti i file devono esistere e non essere vuoti
    missing = []
    for cid in sorted(csv_paths):
        path = csv_paths[cid]
        if not os.path.exists(path):
            print(f"  [✗] client {cid}: file non trovato → {path}")
            missing.append(path)
        elif os.path.getsize(path) == 0:
            print(f"  [✗] client {cid}: file vuoto → {path}")
            missing.append(path)

    if missing:
        print("\n[ERRORE] Dataset mancanti o vuoti per l'analisi per-client:")
        for p in missing:
            print(f"  - {p}")
        print("Esegui prima i controller Ryu per generare i dataset.")
        sys.exit(1)

    results = []
    for cid in sorted(csv_paths):
        path = csv_paths[cid]

        df = pd.read_csv(path).dropna(subset=FEATURE_COLS + [TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

        print(f"\n  client {cid} — {len(df)} campioni")
        dist = df[TARGET_COL].value_counts().sort_index()
        for lbl, cnt in dist.items():
            print(f"    {ALL_CLASSES[lbl]:<8} {cnt:>6}  ({cnt/len(df)*100:.1f}%)")

        res = run_rf_on_client(df, cid)
        results.append(res)

        print(f"    → Acc={res['acc']:.4f}  F1-macro={res['f1_macro']:.4f}")
        print(f"    → Feature più importante: "
              f"{FEATURE_COLS[np.argmax(res['mdi_imp'])]}  "
              f"({res['mdi_imp'].max()*100:.1f}%)")

    if len(results) < 2:
        print("\n[ERRORE] Servono almeno 2 controller per le figure di confronto.")
        sys.exit(1)

    # Salva le 3 figure
    plot_per_client_importance(
        results,
        os.path.join(base_path, "fi_per_controller.png")
    )
    plot_ranking_comparison(
        results,
        os.path.join(base_path, "fi_ranking_comparison.png")
    )
    plot_per_client_cm(
        results,
        os.path.join(base_path, "cm_per_controller.png")
    )

    # Riepilogo testuale
    print("\n" + "="*60)
    print("  RIEPILOGO IMPORTANZA PER CONTROLLER")
    print("="*60)
    header = f"  {'Feature':<14}" + "".join(f"  Ctrl{r['client_id']:>1}" for r in results)
    print(header)
    print("-"*60)
    for feat in FEATURE_COLS:
        idx  = FEATURE_COLS.index(feat)
        vals = "".join(f"  {r['mdi_imp'][idx]*100:>5.1f}%" for r in results)
        print(f"  {feat:<14}{vals}")
    print("="*60)

    return results