# Results summary

All numbers below are copied verbatim from saved notebook outputs. Nothing was recomputed or invented. Each number cites the notebook (original, unredacted path — identical content in `work/redacted/`) and the code-cell index that produced it (0-indexed, counting from the top of the notebook).

## A1 Part 1 — manual feed-forward net on MNIST (`Assignment01_Part1_solution.ipynb`)

Final (epoch 50/50) test accuracy by optimizer:

| Optimizer | Test accuracy | Cell |
|---|---|---|
| SGD | 97.20% | CELL 55 |
| SGD + momentum | 98.05% | CELL 57 |
| AdaGrad | 96.53% | CELL 59 |

(Restated in the notebook's own summary markdown, CELL 60.)

## A1 Part 2 — FashionMNIST MLP, grid search, entropy regularisation, SAM (`Assignment01_Part2_solution.ipynb`)

- **Best-checkpoint test accuracy (Q2.2):** 0.8774, from the checkpoint at epoch 37 (val_acc 0.8862) — CELL 13.
- **12-run architecture grid search** (hidden sizes n1/n2 × activation, selected by **validation** accuracy — legitimate), CELL 17:

| n1 | n2 | activation | best val_acc | epoch |
|---|---|---|---|---|
| 20 | 20 | sigmoid | 0.8727 | 47 |
| 20 | 20 | tanh | 0.8738 | 48 |
| 20 | 20 | relu | 0.8712 | 24 |
| 20 | 40 | sigmoid | 0.8752 | 49 |
| 20 | 40 | tanh | 0.8685 | 28 |
| 20 | 40 | relu | 0.8780 | 36 |
| 40 | 20 | sigmoid | 0.8833 | 49 |
| 40 | 20 | tanh | 0.8787 | 40 |
| 40 | 20 | relu | 0.8858 | 48 |
| 40 | 40 | sigmoid | 0.8830 | 47 |
| 40 | 40 | tanh | 0.8837 | 44 |
| 40 | 40 | **relu** | **0.8875 (winner)** | 41 |

- **Entropy-loss λ sweep** (best-checkpoint **test** accuracy — see caveat below), CELL 20:

| λ | test acc | epoch |
|---|---|---|
| 0.01 | 0.8807 | 26 |
| 0.05 | 0.8790 | 42 |
| 0.1 | 0.8809 | 23 |
| **0.2** | **0.8823 (best)** | 40 |

- **SAM ρ sweep** (best-checkpoint **test** accuracy — see caveat below), CELL 23:

| ρ | best test acc | epoch |
|---|---|---|
| 0.02 | 0.8774 | 38 |
| **0.05** | **0.8838 (best)** | 31 |
| 0.10 | 0.8801 | 40 |

## A1 Part 3 — CNN on 20-class image dataset (`Assignment01_Part3_solution.ipynb`)

Validation accuracy by method, CELLs 32/35/38/41/44/47/48:

| Method | Val acc |
|---|---|
| Baseline CNN | **84.05%** |
| Mixup | 82.68% |
| CutMix | 82.05% |
| OVA (one-vs-all loss) | 82.15% |
| CE + rotation SSL (λ=0.15) | 77.40% |
| OVA + light Mixup | 82.58% |

- **Ensemble** (top-3 by val acc: Baseline, Mixup, OVA+Mixup): val acc = 83.95% (CELL 50).
- **Ensemble gap:** −0.11% — the 3-model ensemble (83.95%) underperforms the single best model, Baseline (84.05%).
- **Per-class accuracy, Baseline model** (CELL 52): hardest classes — `vase` 61.76% (n=34), `bottle` 75.76% (n=33), `cat` 78.00% (n=50), `cake` 79.31% (n=58), `butterfly` 80.39% (n=51). Easiest — `lion` 100.00% (n=38), `fish` 97.83% (n=46).

## A1 Kaggle competition — ConvNeXt-V2 Tiny fine-tune, 20-class animal images (`A1/Kaggle_ConvNeXt_finetune.ipynb`)

- Model: pretrained `convnextv2_tiny.fcmae_ft_in22k_in1k` (timm), fine-tuned at 224×224 (CELL 12).
- Final submission built from a **greedy model soup** (members: TrivialAug run) + 4-view TTA, validation accuracy **83.65%** (CELL 17/22).
- Other ensembling strategies tried but not used: softmax top-2 ensemble 68.01%, 4-model soup 62.23%, top-2 soup 61.45%, top-3 soup 45.81% (CELL 17) — all worse than the single greedy soup.
- No Kaggle public/private leaderboard score is recorded in the notebook (the competition test labels are withheld); the only number tied to the submitted model is the local validation accuracy above.
- *Inconsistency, not resolved:* an earlier "individual val scores" block (CELL 17) reports much higher numbers for the same single models in isolation (TrivialAug 98.52%, CutMix 97.32%, Mixup 89.71%, Baseline 88.51%) before soup/ensembling. This does not match the post-ensembling figures above — likely a different or leakier validation pass used during model selection. Flagged, not resolved; **do not quote the 98.52%/97%+ numbers as the model's real accuracy.**

## A2 — Word embeddings baseline and Text CNN (`...Assignment2_Official[Main].ipynb`)

- **Word-embedding baseline:** Logistic Regression on GloVe (`glove-wiki-gigaword-100`, loaded via gensim; the notebook's own variable is named `word2vect`) sentence vectors — validation accuracy **90.40%** (CELL 52). No separate "Word2Vec" model is trained; this GloVe+LogReg classifier is the only embedding baseline in the notebook.
- **Text CNN:** final-epoch val accuracy 96.46%, test accuracy **97.01%** (test_loss 0.2371) (CELLs 59/61).

## A2 — RNN / LSTM / GRU / Attention runs (`...Assignment2_Official[RNNs].ipynb`)

All runs: `embed_size=128`, `state_sizes=[64,128]`, 30 epochs, Adam lr=1e-3, on the same 198-example validation split.

| Run | Final val_acc | Best val_acc | Cell |
|---|---|---|---|
| SimpleRNN, mean-pool | 91.41% | 94.44% | 28 |
| SimpleRNN, max-pool | 96.46% | 96.46% | 31 |
| **SimpleRNN, last-state** | **25.76%** | **27.27%** | 34 |
| LSTM, mean-pool | 94.95% | 95.96% | 36 |
| LSTM, max-pool | 98.48% | **98.48% (best overall)** | 37 |
| LSTM, last-state | 69.19% | 69.19% | 38 |
| GRU, mean-pool | 95.45% | 95.45% | 40 |
| GRU, max-pool | 96.46% | 97.47% | 41 |
| GRU, last-state | 95.96% | 96.46% | 42 |
| GRU + pretrained GloVe, scratch-init | 97.98% | 97.98% | 48 |
| GRU + pretrained GloVe, frozen (init-only) | 90.40% | 93.43% | 49 |
| GRU + pretrained GloVe, fine-tuned | 97.47% | 97.47% | 50 |
| AttentionRNN (GRU + attention, mean) | 95.96% | 95.96% | 61 |

(Cross-checked against the notebook's own markdown answer in CELL 51, which restates "scratch 97.98%, init-fine-tune 97.47%, init-only 90.40%" — matches exactly.)

## A2 — Prefix-BERT and final best model (`...Assignment2_Official[Transformers]_fixed.ipynb`)

- **Prefix-BERT** (bert-base-uncased, prefix length 5, 100 epochs): final/best printed validation accuracy **72.22%** (CELL 46). This is well below the Text CNN and LSTM-max models.
- **Final best-model comparison** (best-checkpoint, restored by validation accuracy), CELL 48:

| Candidate | Val acc (best ckpt) | Test acc |
|---|---|---|
| **TextCNN (selected)** | **0.9848** | 0.9701 |
| LSTM-max | 0.9747 | 0.9701 |

- **Final best-model test accuracy: 0.9701 (97.01%)**, model = TextCNN from A2 Part 2 (chosen over LSTM-max for higher validation accuracy; both tied on test accuracy). Confirmed in the notebook's own answer markdown, CELL 49.

## `Transformers.ipynb` vs `Transformers_fixed.ipynb`

The two notebooks are identical except for one comment in CELL 48. The non-"fixed" version's comment states "Prefix-BERT reached only 0.8756" — **this number does not appear anywhere in either notebook's saved output** and appears to be a stale/incorrect comment. The `_fixed` version corrects the comment to "~72% validation accuracy," which matches the actual printed output (72.22%, CELL 46) exactly. Both notebooks were kept and both were redacted; only the `_fixed` version's comment is accurate.

## Caveats (read before quoting any number above)

1. **A1 Part 2 SAM/entropy numbers are optimistic.** The entropy-loss λ (CELL 20) and SAM ρ (CELL 23) were both selected by maximum **test-set** accuracy, not validation accuracy — confirmed directly in the code (CELL 20: `max(lam_results, key=lambda r: r[2])  # by test acc of best ckpt`; CELL 23: checkpointing condition `if test_acc >= best_test:` and `best_rho = max(rho_results, key=lambda r: r[1])` on test accuracy; CELL 24 markdown explicitly says checkpoints are saved "by test accuracy"). The reported 0.8823 (λ=0.2) and 0.8838 (ρ=0.05) are therefore inflated relative to a clean val-only selection protocol. The 12-run architecture grid search (CELL 17) is **not** affected — that one is correctly selected on validation accuracy.
2. **A2's validation set has only 198 examples.** Differences of a few points between RNN/GRU/attention variants (e.g. 95.45% vs 97.47%) are within the noise of a 198-example split and should not be read as a reliable ranking.
3. **One RNN run failed to train:** SimpleRNN with `output_type='last_state'` never exceeded ~27% validation accuracy (random chance for this task is ≈16.7%, and training loss also never dropped meaningfully) — it was not tuned further. LSTM with `output_type='last_state'` (69.19%) also clearly underperformed every other LSTM/GRU variant, though it did learn; it likewise was not tuned further.
4. **All results are from single runs with a single seed** (no repeated-seed variance estimates exist in these notebooks). Point differences smaller than roughly 1–2% should not be treated as statistically meaningful.
5. **The Kaggle notebook's "individual val scores" block is inconsistent** with its own final ensembling numbers (see the A1 Kaggle section above) and was not reconciled — only the post-ensembling greedy-soup number (83.65%) is reported as the model's real validation performance.
