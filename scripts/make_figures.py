#!/usr/bin/env python3
"""Generate portfolio figures by parsing saved TEXT outputs already stored in the
notebooks. Never opens a kernel, never re-runs anything — pure regex over
`outputs[].text` in the notebook JSON, then matplotlib for the plotting.

If a series can't be found/parsed reliably, it is skipped and reported, not guessed.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

SKIPPED = []


def load_code_cells(rel_path):
    nb = json.load(open(ROOT / rel_path, encoding="utf-8"))
    return [c for c in nb["cells"] if c.get("cell_type") == "code"]


def cell_text(cell):
    parts = []
    for out in cell.get("outputs", []):
        t = out.get("text")
        if isinstance(t, list):
            parts.append("".join(t))
        elif isinstance(t, str):
            parts.append(t)
    return "\n".join(parts)


def find_series(code_cells, pattern, want_groups=2):
    """Return the regex matches from the first code cell whose output matches the
    pattern at least once. Returns (cell_index, [match_tuples]) or (None, [])."""
    for i, c in enumerate(code_cells):
        text = cell_text(c)
        matches = pattern.findall(text)
        if matches:
            return i, matches
    return None, []


def line_chart(path, title, series, xlabel="Epoch", ylabel="Accuracy (%)"):
    """series: dict label -> (x_values, y_values)"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o" if len(xs) <= 10 else None, markersize=4, label=label, linewidth=1.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def bar_chart(path, title, labels, values, ylabel="Validation accuracy (%)", highlight=None, rotate=30):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#c0392b" if highlight and l in highlight else "#2c7fb8" for l in labels]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.15)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.015, f"{v:.2f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Project 02: MNIST optimiser comparison + FashionMNIST MLP training curve
# ---------------------------------------------------------------------------

def project02():
    out_dir = ROOT / "projects/02-neural-network-training-optimisation/figures"
    code_cells = load_code_cells("projects/02-neural-network-training-optimisation/01_manual_ffn_optimisers.ipynb")

    sgd_pat = re.compile(r"Epoch (\d+)/\d+, Train Loss: [\d.]+, Train Acc: [\d.]+%, Test Loss: [\d.]+, Test Acc: ([\d.]+)%")
    sparse_pat = lambda name: re.compile(name + r" epoch (\d+): train_acc=[\d.]+, test_acc=([\d.]+)")

    series = {}
    i, m = find_series(code_cells, sgd_pat)
    if m:
        series["SGD"] = ([int(e) for e, _ in m], [float(a) for _, a in m])
    else:
        SKIPPED.append("project02/fig01: SGD series not found")

    i, m = find_series(code_cells, sparse_pat("Momentum"))
    if m:
        series["SGD + momentum"] = ([int(e) for e, a in m], [float(a) * 100 for e, a in m])
    else:
        SKIPPED.append("project02/fig01: Momentum series not found")

    i, m = find_series(code_cells, sparse_pat("AdaGrad"))
    if m:
        series["AdaGrad"] = ([int(e) for e, a in m], [float(a) * 100 for e, a in m])
    else:
        SKIPPED.append("project02/fig01: AdaGrad series not found")

    if series:
        line_chart(out_dir / "fig01_mnist_optimiser_comparison.png",
                    "MNIST: Test Accuracy by Optimiser", series)

    code_cells2 = load_code_cells("projects/02-neural-network-training-optimisation/02_mlp_gridsearch_sam.ipynb")
    mlp_pat = re.compile(r"Epoch (\d+)/50: train_acc=([\d.]+) - val_acc=([\d.]+) - test_acc=([\d.]+) - lr=")
    i, m = find_series(code_cells2, mlp_pat)
    if m:
        epochs = [int(e) for e, *_ in m]
        series2 = {
            "Train accuracy": (epochs, [float(t) * 100 for _, t, _, _ in m]),
            "Validation accuracy": (epochs, [float(v) * 100 for _, _, v, _ in m]),
            "Test accuracy": (epochs, [float(te) * 100 for *_, te in m]),
        }
        line_chart(out_dir / "fig02_fashionmnist_mlp_training_curve.png",
                    "FashionMNIST MLP: Training Curve (best architecture, 40×40 ReLU)", series2)
    else:
        SKIPPED.append("project02/fig02: FashionMNIST MLP curve not found")


# ---------------------------------------------------------------------------
# Project 01: CNN baseline training curve, augmentation comparison, per-class accuracy
# ---------------------------------------------------------------------------

def project01():
    out_dir = ROOT / "projects/01-image-classification-cnn/figures"
    code_cells = load_code_cells("projects/01-image-classification-cnn/01_cnn_baseline_and_regularisation.ipynb")

    cnn_pat = re.compile(r"Epoch (\d+)/\d+\s*\ntrain loss= [\d.]+ - train acc= ([\d.]+)% - valid loss= [\d.]+ - valid acc= ([\d.]+)%")
    i, m = find_series(code_cells, cnn_pat)
    if m:
        epochs = [int(e) for e, *_ in m]
        series = {
            "Train accuracy": (epochs, [float(t) for _, t, _ in m]),
            "Validation accuracy": (epochs, [float(v) for *_, v in m]),
        }
        line_chart(out_dir / "fig01_cnn_baseline_training_curve.png",
                    "20-Class CNN Baseline: Training Curve", series)
    else:
        SKIPPED.append("project01/fig01: CNN baseline training curve not found")

    # Augmentation comparison — final validation accuracy per method (RESULTS.md).
    methods = ["Baseline", "Mixup", "CutMix", "OVA", "Rotation SSL", "OVA+Mixup"]
    accs = [84.05, 82.68, 82.05, 82.15, 77.40, 82.58]
    bar_chart(out_dir / "fig02_augmentation_comparison.png",
              "20-Class CNN: Baseline vs Augmentation/Regularisation", methods, accs,
              highlight={"Baseline"}, rotate=20)

    # Per-class accuracy for the baseline model.
    perclass_pat = re.compile(r"^(\w+)\s*:\s*([\d.]+)%\s*\(n=\d+\)", re.M)
    i, m = find_series(code_cells, perclass_pat)
    if m:
        block = cell_text(code_cells[i]).split("Hardest classes:")[0]
        rows = perclass_pat.findall(block)
        rows.sort(key=lambda r: float(r[1]))
        labels = [r[0] for r in rows]
        values = [float(r[1]) for r in rows]
        hardest = {r[0] for r in rows[:5]}
        bar_chart(out_dir / "fig03_per_class_accuracy.png",
                  "20-Class CNN Baseline: Per-Class Validation Accuracy", labels, values,
                  highlight=hardest, rotate=55)
    else:
        SKIPPED.append("project01/fig03: per-class accuracy table not found")


# ---------------------------------------------------------------------------
# Project 03: Text CNN curve, best RNN curve, A2 model comparison bar chart
# ---------------------------------------------------------------------------

BASETRAINER_PAT = re.compile(r"train_accuracy:\s*([\d.]+)%\s*-\s*val_loss:\s*[\d.]+\s*-\s*val_accuracy:\s*([\d.]+)%")


def project03():
    out_dir = ROOT / "projects/03-text-classification-sequence-models/figures"

    main_cells = load_code_cells("projects/03-text-classification-sequence-models/01_word_embeddings_textcnn.ipynb")
    i, m = find_series(main_cells, BASETRAINER_PAT)
    if m:
        epochs = list(range(1, len(m) + 1))
        series = {
            "Train accuracy": (epochs, [float(t) for t, _ in m]),
            "Validation accuracy": (epochs, [float(v) for _, v in m]),
        }
        line_chart(out_dir / "fig01_textcnn_training_curve.png", "Text CNN: Training Curve", series)
    else:
        SKIPPED.append("project03/fig01: Text CNN training curve not found")

    rnn_cells = load_code_cells("projects/03-text-classification-sequence-models/02_rnn_lstm_gru_attention.ipynb")
    # The best RNN run (LSTM, max-pool) is identified by its final validation
    # accuracy matching the headline figure from RESULTS.md (98.48%), not by a
    # fixed cell index, so this still works if cells are added/removed upstream.
    best_cell_idx, best_series = None, None
    for idx, c in enumerate(rnn_cells):
        matches = BASETRAINER_PAT.findall(cell_text(c))
        if matches and abs(float(matches[-1][1]) - 98.48) < 0.05:
            best_cell_idx, best_series = idx, matches
            break
    if best_series:
        epochs = list(range(1, len(best_series) + 1))
        series = {
            "Train accuracy": (epochs, [float(t) for t, _ in best_series]),
            "Validation accuracy": (epochs, [float(v) for _, v in best_series]),
        }
        line_chart(out_dir / "fig02_best_rnn_training_curve.png",
                    "Best RNN Variant (LSTM, max-pool): Training Curve", series)
    else:
        SKIPPED.append("project03/fig02: best RNN (LSTM max-pool) training curve not found (no cell's final val_acc matched 98.48%)")

    # A2 model comparison — final numbers from RESULTS.md, each clearly labelled val/test.
    labels = ["Text CNN\n(test)", "LSTM max-pool\n(val)", "GRU max-pool\n(val)", "Attention RNN\n(val)", "Prefix-BERT\n(val)"]
    values = [97.01, 98.48, 97.47, 95.96, 72.22]
    bar_chart(out_dir / "fig03_model_comparison.png",
              "A2: Model Comparison (Text Classification)", labels, values,
              ylabel="Accuracy (%)", highlight={"Prefix-BERT\n(val)"}, rotate=15)


def main():
    project01()
    project02()
    project03()
    if SKIPPED:
        print("\nSkipped (could not parse reliably):")
        for s in SKIPPED:
            print(" -", s)
    else:
        print("\nAll figures generated; nothing skipped.")


if __name__ == "__main__":
    main()
