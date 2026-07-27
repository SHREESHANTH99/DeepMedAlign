"""
plot_training_graphs.py
-----------------------
Generates professor-quality training graphs from results/training_log.csv.

Outputs (all to results/figures/):
  training_dashboard.png   — 4-panel: total loss, MI loss, Dice loss, Jac loss
  lr_schedule.png          — cosine annealing LR curve
  val_ncc.png              — validation NCC over time
  methods_comparison.png   — bar chart: all methods Dice + HD95
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

df = pd.read_csv("results/training_log.csv")
Path("results/figures").mkdir(parents=True, exist_ok=True)

BG   = "white"
CARD = "#f8f8f8"
C1   = "#2563eb"   # blue    — train
C2   = "#dc2626"   # red     — val
C3   = "#d97706"   # amber   — LR
C4   = "#16a34a"   # green   — dice
C5   = "#9333ea"   # purple  — jac

epochs = df["epoch"]


def styled_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(CARD)
    ax.set_title(title, color="#111111", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color="#444444", fontsize=9)
    ax.set_ylabel(ylabel, color="#444444", fontsize=9)
    ax.tick_params(colors="#333333", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.grid(True, color="#dddddd", alpha=0.8, linewidth=0.5)
    ax.legend(fontsize=8, facecolor="white", edgecolor="#bbbbbb", labelcolor="#111111")


# ============================================================
# Figure 1 — 4-panel Training Dashboard
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
fig.suptitle(
    "DeepMedAlign — VoxelMorph v2 Training Dashboard  (134 epochs, Kaggle T4 GPU)",
    color="white", fontsize=15, fontweight="bold", y=1.01,
)

# Panel 1: Total loss
ax = axes[0, 0]
ax.plot(epochs, df["train_loss"], color=C1, lw=1.8, label="Train Loss")
ax.plot(epochs, df["val_loss"],   color=C2, lw=1.8, label="Val Loss", linestyle="--")
ax.fill_between(epochs, df["train_loss"], df["val_loss"], alpha=0.07, color=C2)
styled_ax(ax, "Total Loss (Train vs Validation)", "Epoch", "Loss")

# Panel 2: MI loss
ax = axes[0, 1]
ax.plot(epochs, df["train_mi"], color=C1, lw=1.8, label="Train MI Loss")
ax.plot(epochs, df["val_mi"],   color=C2, lw=1.8, label="Val MI Loss", linestyle="--")
styled_ax(ax, "Mutual Information Loss", "Epoch", "MI Loss")

# Panel 3: Dice loss
ax = axes[1, 0]
ax.plot(epochs, df["train_dice_loss"], color=C4, lw=1.8, label="Train Dice Loss")
ax.plot(epochs, df["val_dice_loss"],   color=C2, lw=1.8, label="Val Dice Loss", linestyle="--")
styled_ax(ax, "Dice Loss (brain mask alignment)", "Epoch", "Dice Loss")

# Panel 4: Jacobian + Regularisation
ax = axes[1, 1]
ax.plot(epochs, df["train_jac_loss"], color=C5,       lw=1.5, label="Train Jacobian Loss")
ax.plot(epochs, df["train_reg"],      color=C3,       lw=1.5, label="Train Regularisation", linestyle=":")
ax.plot(epochs, df["val_jac_loss"],   color="#c084fc", lw=1.5, label="Val Jacobian Loss",   linestyle="--")
styled_ax(ax, "Jacobian + Regularisation Loss", "Epoch", "Loss")

plt.tight_layout(pad=2.0)
plt.savefig("results/figures/training_dashboard.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: training_dashboard.png")


# ============================================================
# Figure 2 — Learning Rate Schedule
# ============================================================
fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
ax.plot(epochs, df["lr"], color=C3, lw=2, label="Learning Rate")
ax.fill_between(epochs, 0, df["lr"], alpha=0.15, color=C3)
styled_ax(ax, "Cosine Annealing Learning Rate Schedule", "Epoch", "Learning Rate")
plt.tight_layout()
plt.savefig("results/figures/lr_schedule.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: lr_schedule.png")


# ============================================================
# Figure 3 — Validation NCC
# ============================================================
best_ncc = float(df["val_ncc"].max())
fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
ax.plot(epochs, df["val_ncc"], color=C2, lw=2, label="Val NCC")
ax.axhline(y=best_ncc, color=C4, lw=1, linestyle="--",
           label=f"Best NCC = {best_ncc:.4f}")
ax.fill_between(epochs, df["val_ncc"].min(), df["val_ncc"], alpha=0.1, color=C2)
styled_ax(ax, "Validation NCC (MRI-CT Normalised Cross-Correlation)", "Epoch", "NCC")
plt.tight_layout()
plt.savefig("results/figures/val_ncc.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: val_ncc.png")


# ============================================================
# Figure 4 — Methods Comparison Bar Chart
# ============================================================
methods = ["Rigid", "Affine", "B-spline", "VM v1", "VM v2 (ours)"]
dice    = [0.774,   0.775,    0.776,      0.965,   0.9953]
hd95    = [19.5,    19.5,     19.2,       1.22,    0.00]
colors_bar = ["#94a3b8", "#94a3b8", "#94a3b8", "#2563eb", "#16a34a"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle(
    "DeepMedAlign — Methods Comparison on 36 Unseen Test Patients",
    color="#111111", fontsize=14, fontweight="bold",
)

# Dice bar
ax = axes[0]
bars = ax.bar(methods, dice, color=colors_bar, edgecolor="#222244", linewidth=0.8, zorder=3)
ax.set_facecolor(CARD)
ax.set_title("Dice Score  (higher = better)", color="#111111", fontsize=12, fontweight="bold")
ax.set_ylabel("Dice Score", color="#444444", fontsize=9)
ax.tick_params(colors="#333333", labelsize=9)
ax.set_ylim(0.7, 1.03)
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")
ax.grid(True, color="#dddddd", alpha=0.8, linewidth=0.5, axis="y", zorder=0)
ax.axhline(0.776, color=C3, linestyle="--", lw=1.2, label="B-spline baseline (0.776)")
ax.legend(fontsize=8, facecolor="white", edgecolor="#bbbbbb", labelcolor="#111111")
for bar, val in zip(bars, dice):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
            f"{val:.3f}", ha="center", va="bottom",
            color="#111111", fontsize=9, fontweight="bold")

# HD95 bar
ax = axes[1]
bars = ax.bar(methods, hd95, color=colors_bar, edgecolor="#222244", linewidth=0.8, zorder=3)
ax.set_facecolor(CARD)
ax.set_title("HD95  (lower = better)", color="#111111", fontsize=12, fontweight="bold")
ax.set_ylabel("HD95 (mm)", color="#444444", fontsize=9)
ax.tick_params(colors="#333333", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")
ax.grid(True, color="#dddddd", alpha=0.8, linewidth=0.5, axis="y", zorder=0)
for bar, val in zip(bars, hd95):
    label = f"{val:.2f} mm"
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
            label, ha="center", va="bottom",
            color="#111111", fontsize=9, fontweight="bold")

plt.tight_layout(pad=2.0)
plt.savefig("results/figures/methods_comparison.png", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: methods_comparison.png")

print("\nAll 4 graphs generated in results/figures/")
