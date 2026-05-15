"""
Edge Device Latency Simulator for ECG Federated Learning
Models realistic latency, compute time, and energy consumption for medical IoT devices
"""

import numpy as np
from typing import Dict, List, Tuple
import json
import os

class EdgeDeviceSimulator:
    """Simulates heterogeneous edge devices for ECG monitoring"""
    
    # Device types with realistic specifications
    DEVICE_PROFILES = {
        'wearable_patch': {
            'name': 'Wearable ECG Patch',
            'compute_speed': 0.5,      # Relative speed (slow)
            'latency_base_ms': 150,    # Base network latency
            'bandwidth_mbps': 1.0,     # Low bandwidth
            'battery_mah': 200,        # Small battery
            'power_train_mw': 150,     # Power during training
            'power_idle_mw': 5,        # Idle power
            'availability': 0.85,      # Always on body
            'memory_mb': 64
        },
        'bedside_monitor': {
            'name': 'Hospital Bedside Monitor',
            'compute_speed': 2.0,      # Medium speed
            'latency_base_ms': 20,     # Hospital LAN
            'bandwidth_mbps': 100.0,   # High bandwidth
            'battery_mah': 5000,       # Large battery / plugged in
            'power_train_mw': 2000,    # Higher power
            'power_idle_mw': 500,
            'availability': 0.95,      # Always available
            'memory_mb': 512
        },
        'smartphone': {
            'name': 'Patient Smartphone',
            'compute_speed': 1.5,      # Medium-fast
            'latency_base_ms': 80,     # Cellular/WiFi
            'bandwidth_mbps': 20.0,    # Moderate bandwidth
            'battery_mah': 3000,       # Standard phone battery
            'power_train_mw': 800,     # Moderate power
            'power_idle_mw': 100,
            'availability': 0.70,      # Sometimes unavailable
            'memory_mb': 2048
        },
        'gateway': {
            'name': 'Hospital Gateway Server',
            'compute_speed': 5.0,      # Fast
            'latency_base_ms': 5,      # Local network
            'bandwidth_mbps': 1000.0,  # Very high bandwidth
            'battery_mah': float('inf'), # Plugged in
            'power_train_mw': 5000,    # High power
            'power_idle_mw': 1000,
            'availability': 0.99,      # Nearly always available
            'memory_mb': 8192
        }
    }
    
    def __init__(self, num_devices: int = 20):
        self.num_devices = num_devices
        self.devices: List[Dict] = []
        
    def create_device_network(self, device_distribution: Dict[str, int] = None) -> List[Dict]:
        """
        Create a network of heterogeneous edge devices
        
        Args:
            device_distribution: Dict mapping device type to count
                              Default: balanced distribution
        """
        if device_distribution is None:
            # Realistic hospital distribution
            device_distribution = {
                'wearable_patch': 8,      # Many wearables
                'bedside_monitor': 4,     # Some bedside monitors
                'smartphone': 6,          # Patient phones
                'gateway': 2              # Few gateways
            }
        
        device_id = 0
        self.devices = []
        
        for device_type, count in device_distribution.items():
            profile = self.DEVICE_PROFILES[device_type]
            
            for _ in range(count):
                # Add realistic variation to each device
                device = {
                    'id': device_id,
                    'type': device_type,
                    'name': f"{profile['name']}_{device_id}",
                    'compute_speed': profile['compute_speed'] * np.random.uniform(0.8, 1.2),
                    'latency_base_ms': profile['latency_base_ms'] * np.random.uniform(0.7, 1.5),
                    'bandwidth_mbps': profile['bandwidth_mbps'] * np.random.uniform(0.5, 1.0),
                    'battery_mah': profile['battery_mah'],
                    'current_battery_pct': np.random.uniform(30, 100),
                    'power_train_mw': profile['power_train_mw'],
                    'power_idle_mw': profile['power_idle_mw'],
                    'availability': profile['availability'],
                    'memory_mb': profile['memory_mb'],
                    'is_available': np.random.random() < profile['availability']
                }
                
                self.devices.append(device)
                device_id += 1
        
        return self.devices
    
    def simulate_training_round(self, model_size_mb: float = 1.0, 
                               num_samples: int = 100) -> Dict[int, Dict]:
        """
        Simulate one FL training round for all available devices
        
        Args:
            model_size_mb: Size of model parameters in MB
            num_samples: Number of training samples per device
            
        Returns:
            Dict mapping device_id to performance metrics
        """
        results = {}
        
        for device in self.devices:
            if not device['is_available']:
                continue
            
            # Calculate compute time (inversely proportional to compute speed)
            base_compute_time = (num_samples * 0.01) / device['compute_speed']  # seconds
            
            # Calculate communication time
            upload_time = (model_size_mb * 8) / device['bandwidth_mbps']  # seconds
            download_time = upload_time  # Symmetric
            
            # Total latency = compute + communication + network latency
            total_latency = (
                base_compute_time + 
                upload_time + 
                download_time + 
                device['latency_base_ms'] / 1000.0  # Convert to seconds
            )
            
            # Calculate energy consumption
            train_energy = device['power_train_mw'] * base_compute_time / 1000.0  # mWh
            comm_energy = device['power_idle_mw'] * (upload_time + download_time) / 1000.0
            total_energy = train_energy + comm_energy
            
            # Update battery
            battery_drain = (total_energy / device['battery_mah']) * 100 if device['battery_mah'] != float('inf') else 0
            new_battery = max(0, device['current_battery_pct'] - battery_drain)
            
            results[device['id']] = {
                'device_name': device['name'],
                'device_type': device['type'],
                'compute_time_sec': base_compute_time,
                'comm_time_sec': upload_time + download_time,
                'network_latency_sec': device['latency_base_ms'] / 1000.0,
                'total_latency_sec': total_latency,
                'energy_consumed_mwh': total_energy,
                'battery_before_pct': device['current_battery_pct'],
                'battery_after_pct': new_battery,
                'success': True
            }
            
            # Update device battery state
            device['current_battery_pct'] = new_battery
        
        return results
    
    def get_latency_statistics(self) -> Dict:
        """Get statistical summary of device latencies"""
        if not self.devices:
            return {}
        
        latencies = [d['latency_base_ms'] for d in self.devices]
        compute_speeds = [d['compute_speed'] for d in self.devices]
        
        return {
            'num_devices': len(self.devices),
            'available_devices': sum(1 for d in self.devices if d['is_available']),
            'latency_stats': {
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'mean_ms': np.mean(latencies),
                'std_ms': np.std(latencies),
                'median_ms': np.median(latencies)
            },
            'compute_speed_stats': {
                'min': min(compute_speeds),
                'max': max(compute_speeds),
                'mean': np.mean(compute_speeds),
                'std': np.std(compute_speeds)
            },
            'device_type_distribution': self._count_device_types()
        }
    
    def _count_device_types(self) -> Dict[str, int]:
        """Count devices by type"""
        counts = {}
        for device in self.devices:
            dtype = device['type']
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts
    
    def save_device_network(self, path: str = 'results/device_network.json'):
        """Save device network configuration to JSON"""
        stats = self.get_latency_statistics()
        
        output = {
            'statistics': stats,
            'devices': [
                {
                    'id': d['id'],
                    'type': d['type'],
                    'name': d['name'],
                    'latency_ms': d['latency_base_ms'],
                    'compute_speed': d['compute_speed'],
                    'bandwidth_mbps': d['bandwidth_mbps'],
                    'battery_pct': d['current_battery_pct'],
                    'available': d['is_available']
                }
                for d in self.devices
            ]
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return path


if __name__ == '__main__':
    print("Creating edge device network for ECG federated learning...")
    
    simulator = EdgeDeviceSimulator(num_devices=20)
    devices = simulator.create_device_network()
    
    # Print statistics
    stats = simulator.get_latency_statistics()
    print(f"\nDevice Network Statistics:")
    print(f"Total Devices: {stats['num_devices']}")
    print(f"Available Devices: {stats['available_devices']}")
    print(f"\nLatency Distribution:")
    print(f"  Min: {stats['latency_stats']['min_ms']:.1f} ms")
    print(f"  Max: {stats['latency_stats']['max_ms']:.1f} ms")
    print(f"  Mean: {stats['latency_stats']['mean_ms']:.1f} ms")
    print(f"  Std: {stats['latency_stats']['std_ms']:.1f} ms")
    
    print(f"\nDevice Type Distribution:")
    for dtype, count in stats['device_type_distribution'].items():
        print(f"  {dtype}: {count}")
    
    # Simulate one training round
    print("\nSimulating training round (1 MB model, 100 samples)...")
    round_results = simulator.simulate_training_round(model_size_mb=1.0, num_samples=100)
    
    total_latency = sum(r['total_latency_sec'] for r in round_results.values())
    avg_latency = total_latency / len(round_results) if round_results else 0
    total_energy = sum(r['energy_consumed_mwh'] for r in round_results.values())
    
    print(f"\nRound Results ({len(round_results)} devices participated):")
    print(f"  Average Latency: {avg_latency*1000:.1f} ms")
    print(f"  Total Energy: {total_energy:.2f} mWh")
    
    # Save device network
    saved_path = simulator.save_device_network('results/ecg_device_network.json')
    print(f"\nDevice network saved to: {saved_path}")
