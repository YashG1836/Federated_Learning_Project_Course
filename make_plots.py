import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def apply_style() -> None:
	plt.style.use("seaborn-v0_8-whitegrid")
	plt.rcParams.update(
		{
			"font.family": "serif",
			"font.size": 11,
			"axes.titlesize": 13,
			"axes.labelsize": 11,
			"legend.fontsize": 10,
			"figure.titlesize": 14,
			"axes.spines.top": False,
			"axes.spines.right": False,
			"savefig.dpi": 300,
			"savefig.bbox": "tight",
		}
	)


def save_fig(fig: plt.Figure, out_dir: str, filename: str) -> None:
	fig.savefig(os.path.join(out_dir, filename))
	plt.close(fig)


def plot_utk_method_gap_accuracy(out_dir: str) -> None:
	methods = [
		"Baseline",
		"Static\nWeighting",
		"Fairness\nReg.",
		"Adversarial\nFairness",
		"Attack\nBased",
		"DCA-FW",
	]
	fairness_gap = [0.0852, 0.0850, 0.1013, 0.0736, 0.1064, 0.0507]
	accuracy = [0.8730, 0.8500, 0.8550, 0.8550, 0.8785, 0.8730]

	x = np.arange(len(methods))
	width = 0.38

	fig, ax1 = plt.subplots(figsize=(10, 5.5))
	bars1 = ax1.bar(x - width / 2, fairness_gap, width, color="#c0392b", label="Fairness Gap")
	ax1.set_ylabel("Fairness Gap (lower is better)", color="#c0392b")
	ax1.tick_params(axis="y", colors="#c0392b")
	ax1.set_ylim(0, 0.12)

	ax2 = ax1.twinx()
	bars2 = ax2.bar(x + width / 2, accuracy, width, color="#1f618d", alpha=0.9, label="Overall Accuracy")
	ax2.set_ylabel("Overall Accuracy", color="#1f618d")
	ax2.tick_params(axis="y", colors="#1f618d")
	ax2.set_ylim(0.80, 0.90)

	ax1.set_xticks(x)
	ax1.set_xticklabels(methods)
	ax1.set_title("UTKFace: Fairness-Utility Comparison Across Methods")

	handles = [bars1, bars2]
	labels = ["Fairness Gap", "Overall Accuracy"]
	ax1.legend(handles, labels, loc="upper right")

	save_fig(fig, out_dir, "fig_utk_method_gap_accuracy.png")


def plot_utk_racewise_baseline_vs_dcafw(out_dir: str) -> None:
	races = ["White", "Black", "Asian", "Indian", "Others"]
	baseline = [0.885, 0.913, 0.829, 0.915, 0.863]
	dcafw = [0.879, 0.911, 0.860, 0.906, 0.869]

	x = np.arange(len(races))
	width = 0.36

	fig, ax = plt.subplots(figsize=(9.5, 5.2))
	ax.bar(x - width / 2, baseline, width, label="Baseline FL", color="#7f8c8d")
	ax.bar(x + width / 2, dcafw, width, label="DCA-FW", color="#1f618d")
	ax.set_xticks(x)
	ax.set_xticklabels(races)
	ax.set_ylim(0.78, 0.94)
	ax.set_ylabel("Race-wise Accuracy")
	ax.set_title("UTKFace: Race-wise Accuracy (Baseline vs DCA-FW)")
	ax.legend(loc="lower right")

	save_fig(fig, out_dir, "fig_utk_racewise_baseline_vs_dcafw.png")


def plot_fairface_gap_vs_fairness_clients(out_dir: str) -> None:
	fairness_client_ratio = [0, 40, 50, 60]
	fairness_gap = [0.0992, 0.0880, 0.0858, 0.0812]

	fig, ax = plt.subplots(figsize=(9.5, 5.2))
	ax.plot(
		fairness_client_ratio,
		fairness_gap,
		marker="o",
		linewidth=2.4,
		markersize=7,
		color="#c0392b",
	)
	ax.set_xlabel("Fairness Clients (%)")
	ax.set_ylabel("Fairness Gap (lower is better)")
	ax.set_title("FairFace: Fairness Gap vs Fairness-Client Participation")
	ax.set_xticks(fairness_client_ratio)
	ax.set_ylim(0.078, 0.102)

	for x, y in zip(fairness_client_ratio, fairness_gap):
		ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

	save_fig(fig, out_dir, "fig_fairface_gap_vs_fairness_clients.png")


def plot_fairface_accuracy_gap_runs(out_dir: str) -> None:
	runs = ["Baseline", "DCA-FW R1", "DCA-FW R2", "DCA-FW R3"]
	fairness_gap = [0.0992, 0.0880, 0.0858, 0.0812]
	# R3 overall accuracy not explicitly logged in saved output.
	accuracy = [0.7723, 0.7790, 0.7695, np.nan]

	x = np.arange(len(runs))
	width = 0.38

	fig, ax1 = plt.subplots(figsize=(10, 5.5))
	ax1.bar(x - width / 2, fairness_gap, width, color="#c0392b", label="Fairness Gap")
	ax1.set_ylabel("Fairness Gap", color="#c0392b")
	ax1.tick_params(axis="y", colors="#c0392b")
	ax1.set_ylim(0.075, 0.105)

	ax2 = ax1.twinx()
	acc_plot = np.array(accuracy, dtype=float)
	ax2.plot(x, acc_plot, marker="D", linestyle="--", color="#1f618d", linewidth=2, label="Overall Accuracy")
	ax2.set_ylabel("Overall Accuracy", color="#1f618d")
	ax2.tick_params(axis="y", colors="#1f618d")
	ax2.set_ylim(0.765, 0.782)

	ax1.set_xticks(x)
	ax1.set_xticklabels(runs)
	ax1.set_title("FairFace: Accuracy and Fairness-Gap Progression")

	ax2.annotate("Accuracy not logged", xy=(3, 0.7695), xytext=(2.35, 0.7715), fontsize=9, color="#1f618d")

	from matplotlib.lines import Line2D
	from matplotlib.patches import Patch

	legend_items = [Patch(facecolor="#c0392b", label="Fairness Gap"), Line2D([0], [0], color="#1f618d", linestyle="--", marker="D", label="Overall Accuracy")]
	ax1.legend(handles=legend_items, loc="upper right")

	save_fig(fig, out_dir, "fig_fairface_accuracy_gap_runs.png")


def plot_fairface_racewise_baseline_vs_best_dcafw(out_dir: str) -> None:
	races = [
		"White",
		"Black",
		"East Asian",
		"Indian",
		"Latino/Hisp.",
		"Middle East.",
		"SE Asian",
	]
	baseline = [0.7645, 0.7226, 0.7373, 0.8056, 0.8145, 0.8218, 0.7469]
	best_dcafw = [0.7480, 0.7270, 0.7280, 0.7960, 0.8070, 0.8080, 0.7570]

	x = np.arange(len(races))
	width = 0.36

	fig, ax = plt.subplots(figsize=(11, 5.6))
	ax.bar(x - width / 2, baseline, width, label="Baseline FL", color="#7f8c8d")
	ax.bar(x + width / 2, best_dcafw, width, label="Best-gap DCA-FW", color="#1f618d")
	ax.set_xticks(x)
	ax.set_xticklabels(races)
	ax.set_ylim(0.70, 0.84)
	ax.set_ylabel("Race-wise Accuracy")
	ax.set_title("FairFace: Race-wise Accuracy (Baseline vs Best-gap DCA-FW)")
	ax.legend(loc="lower right")

	save_fig(fig, out_dir, "fig_fairface_racewise_baseline_vs_best_dcafw.png")


def main() -> None:
	out_dir = "plots"
	ensure_output_dir(out_dir)
	apply_style()

	plot_utk_method_gap_accuracy(out_dir)
	plot_utk_racewise_baseline_vs_dcafw(out_dir)
	plot_fairface_gap_vs_fairness_clients(out_dir)
	plot_fairface_accuracy_gap_runs(out_dir)
	plot_fairface_racewise_baseline_vs_best_dcafw(out_dir)

	print("Saved plots to:", os.path.abspath(out_dir))
	print("Generated files:")
	for name in sorted(os.listdir(out_dir)):
		print("-", name)


if __name__ == "__main__":
	main()
