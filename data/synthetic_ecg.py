"""Synthetic ECG data generator for federated learning experiments.

This module generates realistic synthetic ECG-like time series data
for benchmarking federated learning compression methods without requiring
external datasets during development and testing.
"""

import numpy as np


def generate_ecg_data(
    samples: int = 600,
    timesteps: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ECG-like time series with binary labels.

    Args:
        samples: Number of samples to generate.
        timesteps: Number of time steps per sample.
        seed: Random seed for reproducibility.

    Returns:
        X: Array of shape (samples, timesteps, 1) containing synthetic ECG signals.
        y: Array of shape (samples,) containing binary labels (0: normal, 1: abnormal).
    """
    rng = np.random.default_rng(seed)
    
    X = np.zeros((samples, timesteps, 1), dtype=np.float32)
    y = np.zeros(samples, dtype=np.int64)
    
    for i in range(samples):
        label = rng.integers(0, 2)
        y[i] = label
        
        t = np.linspace(0, 4 * np.pi, timesteps)
        
        if label == 0:
            signal = _generate_normal_beat(t, rng)
        else:
            signal = _generate_abnormal_beat(t, rng)
        
        noise = rng.normal(0, 0.1, size=timesteps)
        signal = signal + noise
        
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        X[i, :, 0] = signal.astype(np.float32)
    
    return X, y


def _generate_normal_beat(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a normal ECG beat pattern."""
    p_wave = 0.15 * np.exp(-((t - 0.8) ** 2) / 0.15)
    qrs_complex = (
        -0.2 * np.exp(-((t - 1.2) ** 2) / 0.02)
        + 0.5 * np.exp(-((t - 1.25) ** 2) / 0.03)
        - 0.15 * np.exp(-((t - 1.3) ** 2) / 0.02)
    )
    t_wave = 0.25 * np.exp(-((t - 2.0) ** 2) / 0.3)
    
    signal = p_wave + qrs_complex + t_wave
    
    num_cycles = int(4 * np.pi / (2 * np.pi)) + 1
    for cycle in range(1, 3):
        offset = cycle * 2 * np.pi / 3
        signal += (
            0.15 * np.exp(-((t - (0.8 + offset)) ** 2) / 0.15)
            - 0.2 * np.exp(-((t - (1.2 + offset)) ** 2) / 0.02)
            + 0.5 * np.exp(-((t - (1.25 + offset)) ** 2) / 0.03)
            - 0.15 * np.exp(-((t - (1.3 + offset)) ** 2) / 0.02)
            + 0.25 * np.exp(-((t - (2.0 + offset)) ** 2) / 0.3)
        )
    
    return signal


def _generate_abnormal_beat(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate an abnormal ECG beat pattern (arrhythmia-like)."""
    p_wave = 0.1 * np.exp(-((t - 0.6) ** 2) / 0.1)
    qrs_complex = (
        -0.35 * np.exp(-((t - 1.0) ** 2) / 0.04)
        + 0.7 * np.exp(-((t - 1.1) ** 2) / 0.05)
        - 0.25 * np.exp(-((t - 1.2) ** 2) / 0.04)
    )
    t_wave = 0.3 * np.exp(-((t - 1.8) ** 2) / 0.25)
    
    signal = p_wave + qrs_complex + t_wave
    
    arrhythmia_bump = 0.2 * np.sin(3 * t) * np.exp(-((t - 2.5) ** 2) / 0.5)
    signal += arrhythmia_bump
    
    num_cycles = 2
    for cycle in range(1, num_cycles + 1):
        offset = cycle * 2 * np.pi / 2.5
        signal += (
            0.1 * np.exp(-((t - (0.6 + offset)) ** 2) / 0.1)
            - 0.35 * np.exp(-((t - (1.0 + offset)) ** 2) / 0.04)
            + 0.7 * np.exp(-((t - (1.1 + offset)) ** 2) / 0.05)
            - 0.25 * np.exp(-((t - (1.2 + offset)) ** 2) / 0.04)
            + 0.3 * np.exp(-((t - (1.8 + offset)) ** 2) / 0.25)
        )
        signal += 0.15 * np.sin(3 * (t - offset)) * np.exp(-((t - (2.5 + offset)) ** 2) / 0.4)
    
    return signal


if __name__ == "__main__":
    X, y = generate_ecg_data(samples=10, timesteps=200, seed=42)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} timesteps")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Signal shape: {X.shape}, dtype: {X.dtype}")
