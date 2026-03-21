#!/usr/bin/python
"""
Feature Importance Analysis con Random Forest
Dataset: networkdataset_combined.csv  (53.280 pacchetti reali da 3 controller SDN)

Classi target:
  0 = benign
  1 = ack flood
  2 = syn flood
  3 = fin flood
  4 = udp flood

Feature:
  delta_time, eth_type, ip_proto, pkt_len,
  tcp_fin, tcp_syn, tcp_rst, tcp_psh, tcp_ack, tcp_urg
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
base_path   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(base_path, "networkdataset_combined.csv")

FEATURE_COLS = [
    "delta_time", "eth_type", "ip_proto", "pkt_len",
    "tcp_fin", "tcp_syn", "tcp_rst", "tcp_psh", "tcp_ack", "tcp_urg"
]
TARGET_COL  = "target"
CLASS_NAMES = ["benign", "ack", "syn", "fin", "udp"]

PALETTE = {
    "blue":   "#2563EB",
    "orange": "#EA580C",
    "green":  "#16A34A",
    "red":    "#DC2626",
    "gray":   "#6B7280",
}

# ─────────────────────────────────────────────
# 1. CARICAMENTO
# ─────────────────────────────────────────────
print("=== Caricamento dataset combinato ===")
if not os.path.exists(CSV_PATH):
    print(f"\n[ERRORE] File non trovato: {CSV_PATH}")
    print("Esegui prima i controller Ryu e genera il dataset combinato.")
    sys.exit(1)
df = pd.read_csv(CSV_PATH)
if len(df) == 0:
    print(f"\n[ERRORE] File vuoto: {CSV_PATH}")
    sys.exit(1)
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

print(f"Campioni totali : {len(df)}")
print(f"Feature         : {FEATURE_COLS}")
print("\nDistribuzione classi:")
dist = df[TARGET_COL].value_counts().sort_index().rename(index=dict(enumerate(CLASS_NAMES)))
for cls, cnt in dist.items():
    pct = cnt / len(df) * 100
    print(f"  {cls:<8} {cnt:>6}  ({pct:.1f}%)")

# ─────────────────────────────────────────────
# 2. SPLIT & SCALING  (stratificato)
# ─────────────────────────────────────────────
X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. RANDOM FOREST
# ─────────────────────────────────────────────
print("\n=== Addestramento Random Forest (300 alberi) ===")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",   # compensa lo sbilanciamento benign vs flood
)
rf.fit(X_train_sc, y_train)

y_pred = rf.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)
print(f"\nAccuracy test: {acc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 4. TRE MISURE DI FEATURE IMPORTANCE
# ─────────────────────────────────────────────

# 4a. MDI
mdi_imp   = rf.feature_importances_
mdi_std   = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
mdi_order = np.argsort(mdi_imp)[::-1]

# 4b. Permutation (sul test set)
print("Calcolo permutation importance...")
perm_res   = permutation_importance(
    rf, X_test_sc, y_test,
    n_repeats=15, random_state=42, n_jobs=-1, scoring="accuracy"
)
perm_imp   = perm_res.importances_mean
perm_std   = perm_res.importances_std
perm_order = np.argsort(perm_imp)[::-1]

# 4c. Cross-validated MDI (5 fold)
print("Cross-validation feature importance (5-fold)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_importances = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    rf_cv = RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    rf_cv.fit(scaler.fit_transform(X[tr_idx]), y[tr_idx])
    cv_importances.append(rf_cv.feature_importances_)
    print(f"  fold {fold}/5 completato")

cv_arr   = np.array(cv_importances)
cv_mean  = cv_arr.mean(axis=0)
cv_std   = cv_arr.std(axis=0)
cv_order = np.argsort(cv_mean)[::-1]

features_arr = np.array(FEATURE_COLS)

# ─────────────────────────────────────────────
# 5. FIGURA 1 — Feature Importance Dashboard
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#F8FAFC")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.38)

def _bar(ax, order, imp, err, color, bg, xlabel, title, label_color):
    bars = ax.barh(
        features_arr[order], imp[order], xerr=err[order],
        color=color, ecolor=PALETTE["gray"], capsize=4, alpha=0.88
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.set_facecolor(bg)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, imp[order]):
        ax.text(val + max(imp)*0.01, bar.get_y() + bar.get_height()/2,
                f"{val*100:.2f}%", va='center', fontsize=9, color=label_color)

# MDI
_bar(fig.add_subplot(gs[0, 0]),
     mdi_order, mdi_imp, mdi_std,
     PALETTE["blue"], "#EFF6FF",
     "Importanza (MDI)",
     "Feature Importance — MDI\n(Mean Decrease Impurity)",
     "#1E3A5F")

# Permutation
_bar(fig.add_subplot(gs[0, 1]),
     perm_order, perm_imp, perm_std,
     PALETTE["orange"], "#FFF7ED",
     "Calo di Accuracy (Permutation)",
     "Feature Importance — Permutation",
     "#7C2D12")

# CV
_bar(fig.add_subplot(gs[1, 0]),
     cv_order, cv_mean, cv_std,
     PALETTE["green"], "#F0FDF4",
     "Importanza media (5-fold CV)",
     "Feature Importance — 5-Fold CV",
     "#14532D")

# Tabella ranking
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
rank_mdi  = {f: r+1 for r, f in enumerate(features_arr[mdi_order])}
rank_perm = {f: r+1 for r, f in enumerate(features_arr[perm_order])}
rank_cv   = {f: r+1 for r, f in enumerate(features_arr[cv_order])}

table_data = sorted(
    [[f, f"#{rank_mdi[f]}", f"#{rank_perm[f]}", f"#{rank_cv[f]}"] for f in FEATURE_COLS],
    key=lambda r: rank_cv[r[0]]
)
tbl = ax4.table(
    cellText=table_data,
    colLabels=["Feature", "MDI rank", "Perm rank", "CV rank"],
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.55)
for j in range(4):
    tbl[0, j].set_facecolor(PALETTE["blue"])
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(table_data)+1):
    bg = "#EFF6FF" if i % 2 == 0 else "white"
    for j in range(4):
        tbl[i, j].set_facecolor(bg)
ax4.set_title("Confronto Ranking tra i 3 metodi", fontsize=12, fontweight='bold', pad=20)

fig.suptitle(
    f"Random Forest — Feature Importance  |  Dataset combinato ({len(df):,} pacchetti reali)",
    fontsize=14, fontweight='bold', color="#1E293B", y=1.01
)

out1 = os.path.join(base_path, "feature_importance_rf_combined.png")
plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n[✓] Figura 1 salvata: {out1}")

# ─────────────────────────────────────────────
# 6. FIGURA 2 — Confusion Matrix + F1 + distribuzione classi
# ─────────────────────────────────────────────
fig2 = plt.figure(figsize=(18, 6))
fig2.patch.set_facecolor("#F8FAFC")
axes = [fig2.add_subplot(1, 3, i+1) for i in range(3)]

# 6a. Confusion matrix normalizzata
cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=axes[0], colorbar=True, cmap='Blues', values_format='.2f')
axes[0].set_title("Confusion Matrix (normalizzata)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Classe Predetta", fontsize=10)
axes[0].set_ylabel("Classe Reale", fontsize=10)

# 6b. F1 per classe
f1_cls = f1_score(y_test, y_pred, average=None)
bar_colors = [PALETTE["green"] if v >= 0.90 else
              PALETTE["orange"] if v >= 0.70 else
              PALETTE["red"] for v in f1_cls]
axes[1].bar(CLASS_NAMES, f1_cls, color=bar_colors, edgecolor='white', linewidth=1.2, alpha=0.9)
axes[1].set_ylim(0, 1.08)
axes[1].set_ylabel("F1-Score", fontsize=11)
axes[1].set_title("F1-Score per Classe", fontsize=12, fontweight='bold')
axes[1].axhline(0.90, color=PALETTE["gray"], linestyle='--', alpha=0.6, label='Soglia 0.90')
axes[1].legend(fontsize=9)
axes[1].set_facecolor("#F8FAFC")
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
for i, val in enumerate(f1_cls):
    axes[1].text(i, val + 0.018, f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')

# 6c. Distribuzione classi nel dataset (sbilanciamento)
counts = df[TARGET_COL].value_counts().sort_index().values
bar_dist = axes[2].bar(CLASS_NAMES, counts,
                       color=[PALETTE["blue"], PALETTE["orange"], PALETTE["red"],
                               PALETTE["green"], "#7C3AED"],
                       edgecolor='white', linewidth=1.2, alpha=0.88)
axes[2].set_ylabel("N° campioni", fontsize=11)
axes[2].set_title("Distribuzione Classi nel Dataset", fontsize=12, fontweight='bold')
axes[2].set_facecolor("#F8FAFC")
axes[2].grid(axis='y', linestyle='--', alpha=0.4)
for bar, cnt in zip(bar_dist, counts):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                 f"{cnt:,}", ha='center', fontsize=10, fontweight='bold')

fig2.suptitle(
    f"Valutazione Random Forest  |  Accuracy: {acc:.4f}  |  Dataset: {len(df):,} campioni reali",
    fontsize=13, fontweight='bold', color="#1E293B"
)
fig2.tight_layout()

out2 = os.path.join(base_path, "confusion_matrix_rf_combined.png")
plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print(f"[✓] Figura 2 salvata: {out2}")

# ─────────────────────────────────────────────
# 7. RIEPILOGO TESTUALE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  RIEPILOGO FEATURE IMPORTANCE  (dataset combinato reale)")
print("="*60)
print(f"  {'Feature':<14} {'MDI':>8}  {'Perm':>8}  {'CV':>8}")
print("-"*60)
for feat in features_arr[mdi_order]:
    idx = FEATURE_COLS.index(feat)
    print(f"  {feat:<14} {mdi_imp[idx]*100:>7.2f}%  "
          f"{perm_imp[idx]*100:>7.2f}%  "
          f"{cv_mean[idx]*100:>7.2f}%")
print("="*60)
print(f"\n  Accuracy test : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Train         : {len(X_train):,} campioni")
print(f"  Test          : {len(X_test):,} campioni")
print(f"  Sbilanciamento: benign={counts[0]:,}  "
      f"ack={counts[1]:,}  syn={counts[2]:,}  "
      f"fin={counts[3]:,}  udp={counts[4]:,}")
print("="*60)


# ─────────────────────────────────────────────
# 8. ANALISI PER SINGOLO CONTROLLER
# ─────────────────────────────────────────────
from rf_per_controller import run_per_client_analysis

run_per_client_analysis(
    csv_paths={
        1: os.path.join(base_path, "networkdatasetcontroller1.csv"),
        2: os.path.join(base_path, "networkdatasetcontroller2.csv"),
        3: os.path.join(base_path, "networkdatasetcontroller3.csv"),
    },
    base_path=base_path
)