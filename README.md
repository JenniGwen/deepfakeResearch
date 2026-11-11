# Hybrid CNN-Frequency Features for Lightweight Deepfake Detection

> J. G. Tanadi, I. S. Edbert, J. Lim, and S. Ndruru, "Hybrid CNN-Frequency Features for Lightweight Deepfake Detection: An Ablation and Statistically Validated Classifier Comparison," in *Proc. 6th Int. Conf. Intelligent Cybernetics Technology & Applications (ICICyTA)*, Bali, Indonesia, 2026.

The pipeline fine-tunes a CNN backbone on a 4-channel input (RGB + 2D-DFT log-power spectrum), freezes it as a feature extractor, appends two 128-dim handcrafted spectra (azimuthal average and noise-residual spectrum), and compares six classifier heads on the resulting 1536-dim vector.

| | |
|---|---|
| Data | 120,000 balanced face images (60k real / 60k fake), 11 manipulation methods, identity-grouped 70/15/15 split |
| Backbones | EfficientNet-B0 (main), ResNet-50, MobileNetV3-Large, plus an RGB-only EfficientNet-B0 for the ablation |
| Heads | Linear SVM, RBF SVM, Random Forest, Linear + Sigmoid, MLP + Sigmoid, MLP + Softmax |
| Best result | EfficientNet-B0 + MLP + Softmax: **93.32%** accuracy, AUC **0.9769** (18,000 test images) |
| Cost | **~43 ms** per image end to end on 2 CPU cores, ~21 ms with a GPU backbone; 1.6 MB head |

## Downloads
All notebooks, trained checkpoints, extracted features and result tables are in this Google Drive folder:
**https://drive.google.com/drive/folders/16P-niX2ALUdXRlebzgOb2PZBgAOYoqch**

### Notebooks (run in this order)
| # | Notebook | What it does |
|---|---|---|
| 1 | [`ML_deepfake_detector_v6_efficientnetb0.ipynb`](https://colab.research.google.com/drive/1fIK1n7RumXMaXNwxzT7r4loelZKQhu0f) | Builds the leakage-free split, augmentation, 4-channel EfficientNet-B0 fine-tuning, and extracts the frozen features (CNN + AZ + NR). Writes `split_manifest_v6.json`, which every other notebook reuses. |
| 2 | [`01_backbone_resnet50_fixed.ipynb`](https://colab.research.google.com/drive/1pFmc04Fg4wR7JV8_GsvfiVg27UEXvsXt) | Same recipe and split with a ResNet-50 backbone. |
| 3 | [`02_backbone_mobilenetv3_fixed.ipynb`](https://colab.research.google.com/drive/1C1b6PmEytyScQBkXb77nJSJKP4X98MMS) | Same recipe and split with a MobileNetV3-Large backbone. |
| 4 | [`ablation_efnetb0.ipynb`](https://colab.research.google.com/drive/1Dey-o10kdHUNtMh4D-kb0P-01N765JwG)| RGB-only EfficientNet-B0 (no DFT channel, `USE_FFT_CHANNEL = False`), same split, augmentation and schedule. |
| 5 | [`03_classification_stats_efficiency_fixed.ipynb`](https://colab.research.google.com/drive/14ajRghB_AvTYKe3j08CYD7Lpfwmr3j2M) | Six classifier heads, bootstrap CIs, McNemar/DeLong + Holm, feature ablation, 5-fold identity-grouped CV, LOSO, k-NN attribution, latency/FLOPs/memory, and ONNX export. |

### Trained models and outputs

| File | Contents |
|---|---|
| [`deepfake_v6.zip`](https://drive.google.com/file/d/1zhtpyztnPjpOTyYsdJxD2MAQnrfqzt_y/view) | 4-channel EfficientNet-B0: `best_model_v6.pth`, `features_v6.npz`, run log |
| [`deepfake_resnet50.zip`](https://drive.google.com/file/d/1xLY8XEuVp49ET3zNbs19YWIQMRsIAMrS/view) | 4-channel ResNet-50: checkpoint, features, run log |
| [`deepfake_mobilenetv3.zip`](https://drive.google.com/file/d/1GeDd6RIJ3Ojb9ECtYklGclziI-lXxDNS/view) | 4-channel MobileNetV3-Large: checkpoint, features, run log |
| [`deepfake_efficientnet_b0_rgbonly.zip`](https://drive.google.com/file/d/19EBY2V3cd-OKuPgRCvZwAGvo9AB7YF9V/view) | RGB-only EfficientNet-B0 (ablation): checkpoint, features, run log |
| [`deepfake_comparison.zip`](https://drive.google.com/file/d/1kp_NUS-PxIgwl4hCZWjp6zinJ3KvwoJP/view) | All result tables (`tables/paper_tables.xlsx` and CSVs), figures, cached head models, and `deploy/` with the ONNX backbone, ONNX head and `deploy_config.json` |

Each `features_*.npz` holds `X_train/X_val/X_test` (CNN features + 256 handcrafted columns), labels, source and manipulation-type arrays, all in the same row order, so the heads can be retrained without touching the images.

### Dataset

https://www.kaggle.com/datasets/jennifergwen/deepfake-datasets

Built from Celeb-DF v2, the 140k Real and Fake Faces set, DFD frames, FFHQ, two GAN collections and seven DFFD subsets. The notebooks expect these folders under `deepfake_project/` on Google Drive (see `DATASET_PATHS` in the first code cells).

## Environment

All experiments ran on Google Colab with an NVIDIA Tesla T4 GPU and 2 virtual CPU cores.

```bash
pip install torch timm "albumentations>=1.4.0" opencv-python scipy numpy pandas \
            scikit-learn joblib matplotlib seaborn tqdm onnx==1.18.0 onnxruntime
```

Python 3.10, PyTorch 2.x. Seed 42 everywhere.

## Reproducing the paper

1. **Backbones.** Run notebooks 1–4. Each fine-tunes its backbone (3 warm-up epochs, then up to 15 cosine-annealed epochs, early stopping on validation AUC with patience 5) and writes `best_model_<tag>.pth` and `features_<tag>.npz`. Training resumes from checkpoints after a Colab disconnect.
2. **Heads and statistics.** Run notebook 5. Every result is cached to Drive, so it can be re-run after a disconnect and finished cells are skipped. To skip backbone training entirely, unzip the five archives above into `deepfake_project/` and start here.

| Paper item | Notebook 5 cell |
|---|---|
| Table II, Fig. 2 (test metrics, bootstrap CIs) | Cells 5, 6, 9 |
| Table III (McNemar, DeLong, Holm) | Cell 8, Cell 10b |
| Table IV (feature ablation, RGB-only) | Cell 10, Cell 10b |
| 5-fold identity-grouped CV | Cell 11 |
| LOSO (classifier-level) | Cell 12 |
| Per-type detection, k-NN attribution | Cell 13 |
| Table V (backbone cost) | Cell 14 |
| Table VI (end-to-end cost) | Cell 15 |
| ONNX export | Cell 17 |

Each backbone was fine-tuned once, so re-training can shift results by a few tenths of a point.

## Measuring latency

Notebook 5, Cell 14 times each backbone with batch 1 on CPU (fp32, median of 30 runs) and on GPU (fp16, median of 100 runs), plus batch-64 throughput. Cell 15 times preprocessing on 200 test images and adds the head time. On the T4 instance this gives about 9.6 ms preprocessing + 32.8 ms B0 backbone + 0.6 ms MLP head ≈ **43 ms per image**. Latency depends on hardware.
