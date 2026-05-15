# Federated Machine Learning with TurboQuant

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains a comprehensive research framework for evaluating communication-efficient **Federated Learning (FL)** methodologies. It specifically focuses on reducing system latency and communication costs through model compression techniques like **TurboQuant** (vectorized quantization), **Uniform Quantization**, and **Top-K Sparsification**.

The project is currently configured to benchmark these compression methods against standard `FedAvg` using a **CNN-LSTM** model trained on real **ECG (MIT-BIH)** datasets, supplemented by SHAP-based explainability.

## 🌟 Key Features

- **Multi-round FL Simulation**: Robust implementation of federated averaging (`FedAvg`) supporting heterogeneous clients.
- **Advanced Compression**: Includes implementations for:
  - `TurboQuant` (Structured/Vectorized Quantization)
  - `Uniform Quantization`
  - `Top-K Sparsification`
- **Clinical Edge Focus**: Tailored for processing real ECG datasets (MIT-BIH Arrhythmia Database).
- **Comprehensive Benchmarking**: Scripts to evaluate final accuracy, macro-F1, communication costs, latency per round, and accuracy-vs-bits tradeoffs.
- **Explainable AI (XAI)**: Integrated SHAP explainers to interpret federated model decisions.

---

## 📂 Project Structure

```text
.
├── experiments/                 # Benchmarking, statistical analysis, and plotting
│   ├── analyze_benchmark.py
│   ├── compare_compression.py
│   ├── plot_convergence.py
│   ├── real_ecg_mitbih.py
│   ├── recommend.py
│   └── statistical_analysis.py
├── explainability/              # XAI integration
│   └── shap_explainer.py
├── federated/                   # Core FL and Compression logic
│   ├── client.py                # Local training logic
│   ├── server.py                # Aggregation logic
│   ├── experiment.py            # FL orchestrator
│   ├── fedavg.py                # Standard FedAvg
│   ├── turboquant.py            # Vectorized quantization
│   ├── uniform_quant.py         # Baseline scalar quantization
│   └── topk_sparsify.py         # Baseline sparsification
├── model/                       # Neural Network Architectures
│   └── cnn_lstm.py              # CNN-LSTM for time-series (ECG)
├── preprocessing/               # Data pipeline
│   └── download_wfdb.py         # MIT-BIH dataset downloader
├── main.py                      # Primary entry point
└── requirements_*.txt           # Dependency files
```

---

## ⚙️ Installation

To set up the project, it is highly recommended to use a Python virtual environment. The dependencies are split logically to help you install only what you need.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements_core.txt
pip install -r requirements_tff.txt
pip install -r requirements_xai.txt
```

---

## 🚀 Quickstart

**1. Prepare the Data**
Before running experiments, download and preprocess the MIT-BIH dataset:
```bash
python preprocessing/download_wfdb.py
```

**2. Run a Basic FL Experiment**
You can launch the default training pipeline via `main.py`:
```bash
python main.py
```

**3. Benchmark Compression Methods**
To compare uncompressed FedAvg against TurboQuant, Uniform Quantization, and Top-K Sparsification across multiple seeds:
```bash
python experiments/compare_compression.py
```

Results, including statistical analysis and convergence plots, will be generated and saved automatically.

---

## 🔬 Research Directions & Next Steps

This repository serves as a baseline for advanced PhD-level research in Federated Learning. Future extensions planned for this project include:

1. **TurboQuant for Non-IID Data:** Evaluating robustness when client ECG updates are highly heterogeneous.
2. **Adaptive Bit Allocation:** Dynamically adjusting quantization budgets based on client bandwidth, gradient variance, or concept drift.
3. **Privacy-Preserving Aggregation:** Fusing update compression with Differential Privacy (DP) or Secure Aggregation protocols.
4. **Error-Feedback Mechanisms:** Implementing memory buffers across rounds to compensate for quantization errors.

---

## 📝 License

This project is open-sourced under the MIT License.
