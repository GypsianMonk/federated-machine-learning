"""
ECG Federated Learning with Latency Optimization
Compares FedAvg vs LDFL for arrhythmia detection on edge devices
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import json
import os
import time
from datetime import datetime

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulators.ecg_data_generator import ECGDatasetManager
from simulators.edge_device_simulator import EdgeDeviceSimulator


class ECGClassifier(nn.Module):
    """Lightweight 1D CNN for ECG arrhythmia classification on edge devices"""
    
    def __init__(self, input_length: int = 720, num_classes: int = 2):
        super(ECGClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Calculate flattened size after pooling
        self.flat_size = 64 * (input_length // 8)
        
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len) -> (batch, 1, seq_len)
        x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class FederatedClient:
    """Represents a hospital/edge device in the FL network"""
    
    def __init__(self, client_id: int, train_data: Tuple, test_data: Tuple,
                 device_info: Dict):
        self.client_id = client_id
        self.device_info = device_info
        
        # Create data loaders
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        self.train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        self.test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.LongTensor(y_test)
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        # Local model
        self.model = ECGClassifier(input_length=720)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_local(self, epochs: int = 1) -> Dict:
        """Train local model and return metrics"""
        start_time = time.time()
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        train_time = time.time() - start_time
        
        # Evaluate on test set
        accuracy = self.evaluate()
        
        return {
            'client_id': self.client_id,
            'train_time_sec': train_time,
            'avg_loss': total_loss / num_batches,
            'test_accuracy': accuracy,
            'device_type': self.device_info['type'],
            'latency_ms': self.device_info['latency_base_ms']
        }
    
    def evaluate(self) -> float:
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def get_weights(self) -> Dict:
        """Get model weights"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights: Dict):
        """Set model weights"""
        self.model.load_state_dict(weights)


class LDFLCoordinator:
    """Latency-aware Distributed Federated Learning Coordinator"""
    
    def __init__(self, clients: List[FederatedClient], device_simulator: EdgeDeviceSimulator):
        self.clients = clients
        self.device_simulator = device_simulator
        self.global_model = ECGClassifier(input_length=720)
        self.round_history = []
    
    def select_clients_latency_aware(self, selection_ratio: float = 0.6) -> List[int]:
        """
        Select clients based on latency-aware strategy
        Prioritizes low-latency devices while maintaining diversity
        """
        available_clients = [c for c in self.clients if c.device_info['is_available']]
        
        if len(available_clients) == 0:
            return []
        
        # Sort by latency (ascending)
        sorted_clients = sorted(
            available_clients, 
            key=lambda c: c.device_info['latency_base_ms']
        )
        
        # Select top clients by latency
        num_select = max(1, int(len(sorted_clients) * selection_ratio))
        selected = sorted_clients[:num_select]
        
        return [c.client_id for c in selected]
    
    def aggregate_weights(self, selected_client_ids: List[int], 
                         client_metrics: Dict[int, Dict]) -> Dict:
        """
        Aggregate weights with latency-aware weighting
        Lower latency clients get higher weights
        """
        if not selected_client_ids:
            return self.global_model.state_dict()
        
        # Calculate weights based on latency (inverse relationship)
        latencies = []
        for cid in selected_client_ids:
            client = next(c for c in self.clients if c.client_id == cid)
            latencies.append(client.device_info['latency_base_ms'])
        
        # Convert latencies to weights (inverse)
        min_lat = min(latencies)
        weights = [(min_lat / lat) ** 0.5 for lat in latencies]  # Square root for balance
        weights = np.array(weights) / sum(weights)  # Normalize
        
        # Weighted average of parameters
        global_weights = {}
        for key in self.clients[0].get_weights().keys():
            weighted_sum = None
            
            for i, cid in enumerate(selected_client_ids):
                client = next(c for c in self.clients if c.client_id == cid)
                client_weights = client.get_weights()
                
                if weighted_sum is None:
                    weighted_sum = weights[i] * client_weights[key]
                else:
                    weighted_sum += weights[i] * client_weights[key]
            
            global_weights[key] = weighted_sum
        
        return global_weights
    
    def train_round(self, round_num: int, epochs: int = 1) -> Dict:
        """Execute one FL round with latency-aware selection"""
        start_time = time.time()
        
        # Select clients based on latency
        selected_ids = self.select_clients_latency_aware(selection_ratio=0.6)
        
        if not selected_ids:
            return {'round': round_num, 'error': 'No clients available'}
        
        # Distribute global model
        global_weights = self.global_model.state_dict()
        for cid in selected_ids:
            client = next(c for c in self.clients if c.client_id == cid)
            client.set_weights(global_weights)
        
        # Train locally on selected clients
        client_metrics = {}
        for cid in selected_ids:
            client = next(c for c in self.clients if c.client_id == cid)
            metrics = client.train_local(epochs=epochs)
            client_metrics[cid] = metrics
        
        # Aggregate with latency-aware weighting
        new_global_weights = self.aggregate_weights(selected_ids, client_metrics)
        self.global_model.load_state_dict(new_global_weights)
        
        # Evaluate global model
        global_accuracy = self.evaluate_global()
        
        round_time = time.time() - start_time
        
        # Calculate statistics
        avg_latency = np.mean([client_metrics[cid]['latency_ms'] for cid in selected_ids])
        avg_train_time = np.mean([client_metrics[cid]['train_time_sec'] for cid in selected_ids])
        avg_accuracy = np.mean([client_metrics[cid]['test_accuracy'] for cid in selected_ids])
        
        round_result = {
            'round': round_num,
            'selected_clients': len(selected_ids),
            'avg_latency_ms': avg_latency,
            'avg_train_time_sec': avg_train_time,
            'avg_client_accuracy': avg_accuracy,
            'global_accuracy': global_accuracy,
            'round_time_sec': round_time
        }
        
        self.round_history.append(round_result)
        
        return round_result
    
    def evaluate_global(self) -> float:
        """Evaluate global model on all client test sets"""
        total_correct = 0
        total_samples = 0
        
        self.global_model.eval()
        
        with torch.no_grad():
            for client in self.clients:
                for batch_X, batch_y in client.test_loader:
                    outputs = self.global_model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += batch_y.size(0)
                    total_correct += (predicted == batch_y).sum().item()
        
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def train(self, num_rounds: int = 20, epochs_per_round: int = 1) -> Dict:
        """Train for multiple rounds"""
        print(f"\nStarting LDFL Training ({num_rounds} rounds)...")
        
        for round_num in range(num_rounds):
            result = self.train_round(round_num, epochs=epochs_per_round)
            
            if 'error' not in result:
                print(f"Round {round_num+1}/{num_rounds}: "
                      f"Global Acc={result['global_accuracy']:.4f}, "
                      f"Avg Latency={result['avg_latency_ms']:.1f}ms, "
                      f"Time={result['round_time_sec']:.2f}s")
        
        # Final evaluation
        final_accuracy = self.evaluate_global()
        
        return {
            'final_accuracy': final_accuracy,
            'round_history': self.round_history,
            'total_rounds': num_rounds
        }


def run_fedavg_baseline(clients: List[FederatedClient], num_rounds: int = 20) -> Dict:
    """Run standard FedAvg baseline (random client selection)"""
    print(f"\nRunning FedAvg Baseline ({num_rounds} rounds)...")
    
    global_model = ECGClassifier(input_length=720)
    round_history = []
    
    for round_num in range(num_rounds):
        # Random client selection
        available_clients = [c for c in clients if c.device_info['is_available']]
        num_select = max(1, int(len(available_clients) * 0.6))
        selected_clients = np.random.choice(
            available_clients, 
            size=num_select, 
            replace=False
        ).tolist() if len(available_clients) > num_select else available_clients
        
        if not selected_clients:
            continue
        
        # Distribute global model
        global_weights = global_model.state_dict()
        for client in selected_clients:
            client.set_weights(global_weights)
        
        # Train locally
        client_accuracies = []
        latencies = []
        for client in selected_clients:
            metrics = client.train_local(epochs=1)
            client_accuracies.append(metrics['test_accuracy'])
            latencies.append(metrics['latency_ms'])
        
        # Simple averaging (FedAvg)
        avg_weights = {}
        for key in selected_clients[0].get_weights().keys():
            # Only average floating point tensors (skip Long tensors like running_mean in BatchNorm)
            weights_stack = torch.stack([c.get_weights()[key] for c in selected_clients])
            if weights_stack.dtype in [torch.float32, torch.float64]:
                avg_weights[key] = torch.mean(weights_stack, dim=0)
            else:
                # For integer tensors, just take the first one (they're usually identical)
                avg_weights[key] = weights_stack[0]
        
        global_model.load_state_dict(avg_weights)
        
        # Evaluate
        global_model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for client in clients:
                for batch_X, batch_y in client.test_loader:
                    outputs = global_model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += batch_y.size(0)
                    total_correct += (predicted == batch_y).sum().item()
        
        global_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        round_result = {
            'round': round_num,
            'global_accuracy': global_accuracy,
            'avg_latency_ms': np.mean(latencies),
            'avg_client_accuracy': np.mean(client_accuracies)
        }
        
        round_history.append(round_result)
        
        print(f"Round {round_num+1}/{num_rounds}: "
              f"Global Acc={global_accuracy:.4f}, "
              f"Avg Latency={np.mean(latencies):.1f}ms")
    
    return {
        'final_accuracy': global_accuracy,
        'round_history': round_history
    }


def main():
    """Main experiment: ECG FL with latency optimization"""
    print("="*70)
    print("ECG Federated Learning with Latency Optimization")
    print("="*70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Step 1: Generate ECG dataset with Non-IID distribution
    print("\n[1/4] Generating ECG dataset across hospitals...")
    ecg_manager = ECGDatasetManager(num_hospitals=5, samples_per_hospital=500)
    hospital_data = ecg_manager.create_heterogeneous_distribution()
    
    stats = ecg_manager.get_statistics()
    print(f"Created {stats['total_samples']} samples across {stats['num_hospitals']} hospitals")
    print("Non-IID Distribution:")
    for hid, hstats in stats['hospitals'].items():
        print(f"  {hstats['name']}: {hstats['abnormal_ratio']*100:.1f}% abnormal")
    
    # Step 2: Create edge device network
    print("\n[2/4] Creating edge device network...")
    device_sim = EdgeDeviceSimulator()
    devices = device_sim.create_device_network()
    
    dev_stats = device_sim.get_latency_statistics()
    print(f"Created {dev_stats['num_devices']} devices")
    print(f"Latency range: {dev_stats['latency_stats']['min_ms']:.0f}-"
          f"{dev_stats['latency_stats']['max_ms']:.0f} ms")
    
    # Step 3: Create FL clients (one per hospital-device pair)
    print("\n[3/4] Initializing FL clients...")
    clients = []
    
    for hospital_id in range(min(len(hospital_data), len(devices))):
        train_data, test_data = ecg_manager.get_train_test_split(hospital_id, test_ratio=0.2)
        device_info = devices[hospital_id]
        
        client = FederatedClient(
            client_id=hospital_id,
            train_data=train_data,
            test_data=test_data,
            device_info=device_info
        )
        clients.append(client)
    
    print(f"Initialized {len(clients)} FL clients")
    
    # Step 4: Run experiments
    print("\n[4/4] Running FL experiments...")
    
    # Reset clients for fair comparison
    for client in clients:
        client.model = ECGClassifier(input_length=720)
        client.optimizer = optim.Adam(client.model.parameters(), lr=0.001)
    
    # Run FedAvg baseline
    fedavg_results = run_fedavg_baseline(clients.copy(), num_rounds=15)
    
    # Reset clients again
    for client in clients:
        client.model = ECGClassifier(input_length=720)
        client.optimizer = optim.Adam(client.model.parameters(), lr=0.001)
    
    # Run LDFL
    ldfl_coordinator = LDFLCoordinator(clients, device_sim)
    ldfl_results = ldfl_coordinator.train(num_rounds=15, epochs_per_round=1)
    
    # Compare results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    fedavg_final = fedavg_results['final_accuracy']
    ldfl_final = ldfl_results['final_accuracy']
    
    fedavg_latencies = [r['avg_latency_ms'] for r in fedavg_results['round_history']]
    ldfl_latencies = [r['avg_latency_ms'] for r in ldfl_results['round_history']]
    
    avg_fedavg_latency = np.mean(fedavg_latencies)
    avg_ldfl_latency = np.mean(ldfl_latencies)
    
    print(f"\nFinal Global Accuracy:")
    print(f"  FedAvg:  {fedavg_final:.4f}")
    print(f"  LDFL:    {ldfl_final:.4f}")
    print(f"  Improvement: {(ldfl_final - fedavg_final)*100:.2f}%")
    
    print(f"\nAverage Latency per Round:")
    print(f"  FedAvg:  {avg_fedavg_latency:.1f} ms")
    print(f"  LDFL:    {avg_ldfl_latency:.1f} ms")
    print(f"  Reduction: {(avg_fedavg_latency - avg_ldfl_latency)/avg_fedavg_latency*100:.1f}%")
    
    # Save results
    results = {
        'experiment': 'ecg_federated_learning',
        'timestamp': datetime.now().isoformat(),
        'dataset': stats,
        'devices': dev_stats,
        'fedavg': {
            'final_accuracy': fedavg_final,
            'avg_latency_ms': avg_fedavg_latency,
            'round_history': fedavg_results['round_history']
        },
        'ldfl': {
            'final_accuracy': ldfl_final,
            'avg_latency_ms': avg_ldfl_latency,
            'round_history': ldfl_results['round_history']
        },
        'improvements': {
            'accuracy_gain': ldfl_final - fedavg_final,
            'latency_reduction_pct': (avg_fedavg_latency - avg_ldfl_latency) / avg_fedavg_latency * 100
        }
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/ecg_fl_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results/ecg_fl_results.json")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()
