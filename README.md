# Deep Learning Portfolio

Three deep learning projects spanning from-scratch optimisation, CNN image classification, and NLP sequence models — covering manual backpropagation, systematic augmentation/regularisation comparisons, a Kaggle-style fine-tuning pipeline, and a progression from word embeddings through RNNs/attention to a prefix-tuned transformer.

## Projects

| Project | Techniques | Headline result | Link |
|---|---|---|---|
| Neural Network Training & Optimisation | Manual backprop, SGD/momentum/AdaGrad, architecture grid search, entropy regularisation, Sharpness-Aware Minimisation (SAM) | 98.05% test accuracy (MNIST, momentum SGD); 88.75% val accuracy (FashionMNIST MLP) | [`projects/02-neural-network-training-optimisation/`](projects/02-neural-network-training-optimisation/) |
| Image Classification with CNNs | CNN from scratch, Mixup, CutMix, one-vs-all loss, rotation self-supervision, model ensembling, ConvNeXt-V2 fine-tuning, model-soup + TTA | 84.05% val accuracy (CNN baseline); 83.65% val accuracy (ConvNeXt-V2 Kaggle fine-tune) | [`projects/01-image-classification-cnn/`](projects/01-image-classification-cnn/) |
| Text Classification & Sequence Models | GloVe embeddings, Text CNN, SimpleRNN/LSTM/GRU, pooling strategies, attention, prefix-tuned BERT | 97.01% test accuracy (Text CNN, final selected model); 98.48% val accuracy (best RNN variant) | [`projects/03-text-classification-sequence-models/`](projects/03-text-classification-sequence-models/) |

Full results tables with per-notebook citations: [`docs/RESULTS.md`](docs/RESULTS.md).

## Skills & tools demonstrated

- **Deep learning fundamentals:** manual forward/backward-pass derivation and implementation (no autograd shortcuts), gradient-based optimisation (SGD, momentum, AdaGrad, Adam).
- **Regularisation & generalisation:** entropy-regularised loss, Sharpness-Aware Minimisation (SAM), Mixup, CutMix, a one-vs-all loss, rotation self-supervision, model ensembling.
- **Computer vision:** CNNs built from scratch, fine-tuning a pretrained ConvNeXt-V2 backbone, test-time augmentation, model-soup ensembling.
- **NLP:** word embeddings (GloVe), a Text CNN, RNN/LSTM/GRU architectures with multiple pooling strategies, an attention mechanism, prefix-tuning a pretrained BERT model.
- **Experimentation practice:** architecture grid search, hyperparameter sweeps, per-class error analysis, and — in `docs/RESULTS.md` — an honest accounting of where a sweep's selection protocol was flawed (test-set rather than validation-set selection) rather than hiding it.
- **Tooling:** Python, PyTorch, Hugging Face `transformers`, `timm`, scikit-learn, gensim, trained on Apple MPS (Apple Silicon GPU).

Author: Rahul Bhatt
*Built as part of Monash University FIT5215 Deep Learning (2026).*
