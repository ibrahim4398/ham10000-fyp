# Skin Lesion Classification on HAM10000 — Methodological Evaluation

Final-year project, BSc Computer Science with Artificial Intelligence (CHP2524, University of Huddersfield, 2025/26).

A controlled methodological evaluation of skin lesion classification on the HAM10000 dataset, addressing five recurring weaknesses in the published literature: lesion-level data leakage, unjustified imbalance correction, training-regime opacity, the universality assumption across paradigms, and explainability validated only on in-distribution data.

## Notebooks

| Notebook | Phase | Output |
|---|---|---|
| `1A_phase_I_imbalance.ipynb` | Phase I — Imbalance strategy comparison | WeightedRandomSampler selection |
| `1B_scratch_cnn_ladder.ipynb` | Phase II — Scratch CNN ladder B0→B4 | Best scratch checkpoint (B4) |
| `2_transfer_learning.ipynb` | Phase III — Transfer ladder B0→B4 | Best transfer checkpoint (B2) |
| `3_hybrid_gradcam.ipynb` | Phase IV — Hybrid ensemble + Grad-CAM | Hybrid α=0.20 + figures |

## How to reproduce

1. Get the HAM10000 dataset from [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T).
2. Open the notebooks in Google Colab.
3. Run order: 1A → 1B → 2 → 3.
4. Seed is fixed at 42 throughout; deterministic CUDA flags are enabled.

## Key results (leakage-free test partition, n = 1,527)

| Model | Test acc | Test bacc | Test F1 |
|---|---|---|---|
| Scratch CNN (B4) | 0.6876 | 0.6163 | 0.7148 |
| Transfer ResNet50 (B2) | 0.8251 | 0.7789 | 0.8242 |
| Hybrid (α = 0.20) | 0.8265 | 0.7839 | 0.8258 |

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
Pillow>=10.0
```

## Author

Ibrahim Kerouaz (U2290136)
Supervised by Dr Muhammad Hussain and Dr Abirami Gunasekaran
School of Computing and Engineering, University of Huddersfield
