# Explainable Recommendation in Graph Neural Networks
### Propagation Path Analysis and Counterfactual Graph Editing on LightGCN

> Capstone project by **Pranav Nallari**, Vellore Institute of Technology  
> Dataset: MovieLens-1M | Model: LightGCN (3-layer, 64-dim, BPR Loss)

---

## What This Project Does

This project adds explainability to LightGCN-based recommendation systems. Given a recommendation, the system identifies **which training interactions most influenced it** and **at which graph propagation depth** the signal originated.

Two methods are implemented:
- **Propagation Path Analysis** — decomposes recommendation scores layer by layer to attribute influence to specific training items
- **Counterfactual Graph Editing** — removes edges from the interaction graph and measures the resulting score drop to verify attribution

Results are evaluated against three baselines (Random, Degree, LIME) with a faithfulness evaluation against post-fine-tuning ground truth.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional but recommended)

Install dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib seaborn jupyter
```

---

## Dataset Setup

1. Download MovieLens-1M from [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)
2. Extract the zip and place the files inside `dataset/`:

```
dataset/ratings.dat
dataset/movies.dat
```

The data loader handles all preprocessing (5-core filtering, binarisation, 80/20 split) automatically.

---

## Training

To train the LightGCN model from scratch:

```bash
python src/main.py
```

Default configuration (edit inside `main.py` to change):

| Parameter | Value |
|-----------|-------|
| Layers (K) | 3 |
| Embedding dim | 64 |
| Learning rate | 1e-3 |
| L2 regularisation | 1e-4 |
| Batch size | 2048 |
| Epochs | 100 |
| Loss | BPR |

Trained model weights are saved automatically. Training takes approximately 15–25 minutes on GPU.

---

## Running the Explainer

After training, run the explainability pipeline:

```bash
python src/explainer.py
```

This will:
1. Load the trained model
2. Run Propagation Path Analysis for sampled users
3. Run Counterfactual Graph Editing at ranks 1, 5, 10, 20
4. Compare against Random, Degree, and LIME baselines
5. Run faithfulness evaluation (fine-tunes for 5 epochs per edge removal)
6. Save all results to `results.txt`

> ⚠️ The faithfulness evaluation is slow — it runs 5 epochs of fine-tuning per edge removal across 50 users × 3 edges = 150 fine-tuning runs. Expect 1–2 hours on GPU.

---

## Generating Figures

All paper figures are generated from the Jupyter notebook:

```bash
jupyter notebook visualizations_full.ipynb
```

Run all cells top to bottom. Figures are saved to the `figures/` directory as both `.png` and `.pdf`. The notebook is fully self-contained — all data is hardcoded from `results.txt` so no model needs to be loaded.

Generated figures:

| File | Description |
|------|-------------|
| `fig1_architecture.png` | System pipeline diagram |
| `fig2_training_loss.png` | BPR loss over 100 epochs |
| `fig3_degree_distributions.png` | User and item degree histograms |
| `fig4_score_drop_bar.png` | Attribution quality bar chart |
| `fig5_score_drop_boxplot.png` | Score drop distributions |
| `fig6_rank_change_rate.png` | Rank change rate line chart |
| `fig7_faithfulness_scatter.png` | Predicted vs actual scatter |
| `fig8_faithfulness_error.png` | Faithfulness error histogram |
| `fig9_layer_heatmap.png` | Layer contribution heatmap |
| `fig10_layer_bar.png` | Layer contribution grouped bars |
| `fig11_per_user_heatmap.png` | Per-user score drop comparison |
| `fig12_case_study.png` | User 5439 case study |

---

## Key Results

| Method | Mean \|Score Drop\| | vs. Random |
|--------|---------------------|------------|
| **PropPath (Ours)** | **0.0322** | **3.02×** |
| LIME | 0.0255 | 2.38× |
| Random | 0.0107 | 1.00× |
| Degree | 0.0060 | 0.56× |

Faithfulness (structural sensitivity vs. post-fine-tuning): Pearson r = 0.536



---