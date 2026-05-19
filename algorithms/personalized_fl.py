"""
Personalized Federated Learning for ECG
Implements pFedMe (Personalized Federated Learning with Moreau Envelopes)
for patient-specific arrhythmia detection
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class pFedMeOptimizer:
    """
    Personalized Federated Learning optimizer using Moreau Envelopes.
    Allows each client to maintain personalized model parameters while
    benefiting from global knowledge.
    """
    
    def __init__(self, model: nn.Module, lr: float = 0.01, 
                 lambda_reg: float = 0.1, k_steps: int = 5):
        self.model = model
        self.lr = lr
        self.lambda_reg = lambda_reg  # Regularization strength
        self.k_steps = k_steps  # Local update steps
        
        # Store personalized and global parameters
        self.personalized_params = {k: v.clone() for k, v in model.named_parameters()}
        self.global_params = {k: v.clone() for k, v in model.named_parameters()}
        
    def local_update(self, data_loader, device: str) -> Dict:
        """
        Perform k steps of local personalized updates.
        
        Args:
            data_loader: Local data loader
            device: Computing device
            
        Returns:
            Dictionary with update statistics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self.k_steps):
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Forward pass with personalized params
                output = self.model(batch_x)
                loss_fn = nn.CrossEntropyLoss()
                base_loss = loss_fn(output, batch_y)
                
                # Add regularization toward global params
                reg_loss = 0.0
                for name, param in self.model.named_parameters():
                    if name in self.global_params:
                        reg_loss += torch.norm(param - self.global_params[name])**2
                
                total_loss_batch = base_loss + self.lambda_reg * reg_loss
                
                # Backward pass
                self.model.zero_grad()
                total_loss_batch.backward()
                
                # Update personalized params
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param.data -= self.lr * param.grad
                
                total_loss += total_loss_batch.item()
                num_batches += 1
        
        # Update stored personalized params
        self.personalized_params = {k: v.clone() for k, v in self.model.named_parameters()}
        
        return {
            'average_loss': total_loss / max(num_batches, 1),
            'num_batches': num_batches
        }
    
    def update_global_params(self, new_global: Dict[str, torch.Tensor]):
        """Update global parameters from server aggregation"""
        self.global_params = {k: v.clone() for k, v in new_global.items()}
        
        # Soft interpolation between personalized and global
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.global_params:
                    # Move personalized params slightly toward global
                    alpha = 0.3  # Interpolation factor
                    param.data = (1 - alpha) * param.data + alpha * self.global_params[name]
    
    def get_personalized_params(self) -> Dict[str, torch.Tensor]:
        """Return current personalized parameters"""
        return {k: v.clone() for k, v in self.model.named_parameters()}
    
    def get_parameter_divergence(self) -> float:
        """Measure how much personalized params diverge from global"""
        total_divergence = 0.0
        count = 0
        
        for name, param in self.model.named_parameters():
            if name in self.global_params:
                divergence = torch.norm(param - self.global_params[name]).item()
                total_divergence += divergence
                count += 1
        
        return total_divergence / max(count, 1)


def demonstrate_personalized_fl():
    """Demonstrate pFedMe on synthetic ECG classification"""
    print("="*60)
    print("PERSONALIZED FL DEMONSTRATION (pFedMe)")
    print("="*60)
    
    # Simple CNN for ECG
    class SimpleECGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
            self.fc = nn.Linear(16, 2)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = x.mean(dim=2)  # Global pooling
            return self.fc(x)
    
    model = SimpleECGNet()
    
    # Create pFedMe optimizer
    optimizer = pFedMeOptimizer(model, lr=0.01, lambda_reg=0.1, k_steps=3)
    
    print("\nInitializing personalized FL...")
    print(f"  Regularization λ: {optimizer.lambda_reg}")
    print(f"  Local steps k: {optimizer.k_steps}")
    
    # Simulate initial global model
    initial_global = {k: v.clone() for k, v in model.named_parameters()}
    
    # Simulate local training on "patient-specific" data
    print("\nSimulating local training on patient data...")
    
    # Fake data loader
    class FakeDataLoader:
        def __iter__(self):
            for _ in range(5):
                x = torch.randn(4, 1, 10)  # 4 samples, 1 channel, 10 timesteps
                y = torch.randint(0, 2, (4,))
                yield x, y
    
    stats = optimizer.local_update(FakeDataLoader(), device='cpu')
    
    print(f"\nAfter local update:")
    print(f"  Average loss: {stats['average_loss']:.4f}")
    print(f"  Batches processed: {stats['num_batches']}")
    
    # Simulate receiving updated global model
    print("\nReceiving aggregated global model from server...")
    new_global = {k: v + 0.01 for k, v in initial_global.items()}
    optimizer.update_global_params(new_global)
    
    # Check divergence
    divergence = optimizer.get_parameter_divergence()
    print(f"\nParameter divergence from global: {divergence:.6f}")
    
    personalized = optimizer.get_personalized_params()
    print(f"\nPersonalized model ready for patient-specific inference")
    print(f"  Total parameters: {sum(p.numel() for p in personalized.values())}")
    
    return {
        'loss': stats['average_loss'],
        'divergence': divergence,
        'num_params': sum(p.numel() for p in personalized.values())
    }


if __name__ == "__main__":
    demonstrate_personalized_fl()
