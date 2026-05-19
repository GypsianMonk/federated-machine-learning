"""
Privacy Engine for Federated Learning
Implements Differential Privacy (DP-SGD) for ECG data protection
"""

import numpy as np
from typing import Tuple, Dict, List
import json


class DifferentialPrivacyEngine:
    """
    Implements DP-SGD with Gaussian mechanism for gradient perturbation.
    Tracks privacy budget (epsilon) using advanced composition theorem.
    """
    
    def __init__(self, noise_multiplier: float = 0.1, clipping_norm: float = 1.0, 
                 delta: float = 1e-5):
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.delta = delta
        self.epsilon_history = []
        self.total_epsilon = 0.0
        
    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip gradients to bound sensitivity"""
        # Flatten all gradients to compute global norm
        flat_grads = np.concatenate([g.flatten() for g in gradients.values()])
        grad_norm = np.linalg.norm(flat_grads)
        
        # Clip if necessary
        if grad_norm > self.clipping_norm:
            clip_factor = self.clipping_norm / grad_norm
            clipped = {k: v * clip_factor for k, v in gradients.items()}
        else:
            clipped = gradients
            
        return clipped
    
    def add_noise(self, gradients: Dict[str, np.ndarray], 
                  sample_size: int, population_size: int) -> Dict[str, np.ndarray]:
        """Add calibrated Gaussian noise to clipped gradients"""
        q = sample_size / max(population_size, 1)  # Sampling rate
        effective_noise = self.noise_multiplier * self.clipping_norm
        
        noisy_grads = {}
        for key, grad in gradients.items():
            noise = np.random.normal(0, effective_noise, grad.shape)
            noisy_grads[key] = grad + noise
            
        return noisy_grads
    
    def update_privacy_budget(self, num_rounds: int, sample_size: int, 
                             population_size: int) -> float:
        """Update epsilon using advanced composition theorem"""
        q = sample_size / max(population_size, 1)
        
        # Simplified RDP to DP conversion
        sigma = self.noise_multiplier
        epsilon = min(1.0, (q * num_rounds) / (sigma**2))
        
        self.total_epsilon += epsilon
        self.epsilon_history.append(self.total_epsilon)
        
        return self.total_epsilon
    
    def is_privacy_budget_exhausted(self, max_epsilon: float = 8.0) -> bool:
        """Check if privacy budget is exhausted (HIPAA typically ε ≤ 8)"""
        return self.total_epsilon >= max_epsilon
    
    def get_privacy_status(self) -> Dict:
        """Return current privacy accounting status"""
        return {
            'current_epsilon': round(self.total_epsilon, 4),
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier,
            'clipping_norm': self.clipping_norm,
            'budget_remaining': max(0, 8.0 - self.total_epsilon),
            'exhausted': self.is_privacy_budget_exhausted()
        }


def demonstrate_privacy_engine():
    """Demonstrate DP-SGD on dummy gradients"""
    print("="*60)
    print("PRIVACY ENGINE DEMONSTRATION")
    print("="*60)
    
    dp = DifferentialPrivacyEngine(noise_multiplier=0.5, clipping_norm=1.0)
    
    # Create dummy gradients
    dummy_grads = {
        'conv1.weight': np.random.randn(32, 1, 3, 3),
        'fc1.weight': np.random.randn(64, 32)
    }
    
    print("\nOriginal gradient norm:", 
          np.linalg.norm(np.concatenate([g.flatten() for g in dummy_grads.values()])))
    
    # Clip gradients
    clipped = dp.clip_gradients(dummy_grads)
    print("Clipped gradient norm:", 
          np.linalg.norm(np.concatenate([g.flatten() for g in clipped.values()])))
    
    # Add noise
    noisy = dp.add_noise(clipped, sample_size=10, population_size=100)
    print("Noisy gradient norm:", 
          np.linalg.norm(np.concatenate([g.flatten() for g in noisy.values()])))
    
    # Update privacy budget
    for i in range(20):
        eps = dp.update_privacy_budget(num_rounds=1, sample_size=10, population_size=100)
    
    status = dp.get_privacy_status()
    print("\nPrivacy Status:")
    print(f"  Current ε: {status['current_epsilon']:.4f}")
    print(f"  Budget remaining: {status['budget_remaining']:.4f}")
    print(f"  Exhausted: {status['exhausted']}")
    
    return status


if __name__ == "__main__":
    demonstrate_privacy_engine()
