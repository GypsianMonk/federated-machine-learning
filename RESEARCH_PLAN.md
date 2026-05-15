# TurboQuant for Federated Learning: Research Path

This repository is now a baseline, not yet a PhD contribution.

## What this baseline supports

- Multi-round federated learning experiments
- Uncompressed `FedAvg` comparison against TurboQuant-compressed updates
- Communication-aware reporting with compression ratio and bits per value
- Held-out evaluation with accuracy, macro-F1, and log-loss

## What is still missing for PhD-level work

- A faithful implementation of the full TurboQuant method from the paper
- Real ECG datasets instead of only synthetic data
- Comparisons against stronger communication baselines
- Repeated trials with confidence intervals
- Statistical testing and ablation studies
- Convergence analysis or a novel FL-specific extension

## Strong thesis directions

1. TurboQuant for non-IID federated learning
Study whether TurboQuant behaves differently when client updates are highly heterogeneous.

2. TurboQuant with privacy constraints
Combine update compression with differential privacy or secure aggregation.

3. Adaptive TurboQuant in FL
Change the bit-width or quantization budget based on client bandwidth, drift, or gradient variance.

4. Clinical edge deployment
Benchmark communication savings and diagnostic accuracy for ECG or arrhythmia detection workloads on realistic edge settings.

## Immediate next experiments

1. Run `python experiments/compare_compression.py`
2. Repeat with 3 to 5 random seeds
3. Add a real ECG dataset split across clients
4. Compare against at least:
   - FP32 FedAvg
   - uniform scalar quantization
   - top-k sparsification
5. Report:
   - final accuracy
   - macro-F1
   - communication cost
   - latency per round
   - accuracy-vs-bits tradeoff

## Minimal publishable claim template

"TurboQuant-inspired structured quantization reduces FL communication cost by Xx while preserving Y% of predictive performance under non-IID ECG classification."

That is still not enough by itself. For a stronger paper, add a new method:

- adaptive bit allocation per client
- error-feedback across rounds
- robustness to client drift
- privacy-aware compressed aggregation
