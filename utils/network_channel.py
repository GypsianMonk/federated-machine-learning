"""
Network Channel Simulator for Federated Learning
Simulates realistic network conditions: packet loss, jitter, bandwidth throttling
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time


class NetworkChannelSimulator:
    """
    Simulates realistic network conditions for edge devices in FL.
    Models packet loss, latency jitter, and bandwidth constraints.
    """
    
    def __init__(self, packet_loss_rate: float = 0.02, 
                 jitter_std: float = 0.1,
                 bandwidth_mbps: float = 10.0):
        self.packet_loss_rate = packet_loss_rate  # 0-1
        self.jitter_std = jitter_std  # Standard deviation of latency
        self.bandwidth_mbps = bandwidth_mbps
        
        self.packets_sent = 0
        self.packets_lost = 0
        self.total_latency = 0.0
        
    def simulate_transmission(self, data_size_mb: float, 
                             base_latency_ms: float) -> Dict:
        """
        Simulate transmitting data over the network channel.
        
        Args:
            data_size_mb: Size of data to transmit (MB)
            base_latency_ms: Base round-trip latency (ms)
            
        Returns:
            Dictionary with transmission results
        """
        self.packets_sent += 1
        
        # Check for packet loss
        if np.random.random() < self.packet_loss_rate:
            self.packets_lost += 1
            return {
                'success': False,
                'reason': 'packet_loss',
                'latency_ms': None,
                'effective_bandwidth_mbps': 0
            }
        
        # Calculate transmission time with jitter
        transmission_time = (data_size_mb * 8) / max(self.bandwidth_mbps, 0.1)  # seconds
        transmission_time_ms = transmission_time * 1000
        
        # Add jitter to latency
        jitter = np.random.normal(0, self.jitter_std * base_latency_ms)
        total_latency = base_latency_ms + transmission_time_ms + jitter
        total_latency = max(total_latency, 1.0)  # Minimum 1ms
        
        self.total_latency += total_latency
        
        effective_bw = (data_size_mb * 8) / (total_latency / 1000)  # Mbps
        
        return {
            'success': True,
            'reason': None,
            'latency_ms': round(total_latency, 2),
            'effective_bandwidth_mbps': round(effective_bw, 2),
            'jitter_ms': round(jitter, 2)
        }
    
    def get_channel_statistics(self) -> Dict:
        """Return current channel statistics"""
        loss_rate_actual = self.packets_lost / max(self.packets_sent, 1)
        avg_latency = self.total_latency / max(self.packets_sent - self.packets_lost, 1)
        
        return {
            'packets_sent': self.packets_sent,
            'packets_lost': self.packets_lost,
            'actual_loss_rate': round(loss_rate_actual, 4),
            'configured_loss_rate': self.packet_loss_rate,
            'average_latency_ms': round(avg_latency, 2),
            'bandwidth_mbps': self.bandwidth_mbps,
            'jitter_std_ms': round(self.jitter_std * 100, 2)  # Approximate
        }
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self.packets_sent = 0
        self.packets_lost = 0
        self.total_latency = 0.0


def demonstrate_network_simulator():
    """Demonstrate network channel simulation"""
    print("="*60)
    print("NETWORK CHANNEL SIMULATOR DEMONSTRATION")
    print("="*60)
    
    # Create simulator with realistic hospital WiFi conditions
    channel = NetworkChannelSimulator(
        packet_loss_rate=0.05,  # 5% packet loss
        jitter_std=0.15,         # 15% jitter
        bandwidth_mbps=5.0       # 5 Mbps (constrained medical IoT)
    )
    
    print("\nSimulating 50 ECG model transmissions...")
    print(f"Configured: {channel.packet_loss_rate*100}% loss, "
          f"{channel.bandwidth_mbps} Mbps")
    
    latencies = []
    successes = 0
    
    for i in range(50):
        # Simulate sending a 2MB model update
        result = channel.simulate_transmission(data_size_mb=2.0, base_latency_ms=50.0)
        
        if result['success']:
            successes += 1
            latencies.append(result['latency_ms'])
        else:
            print(f"  Round {i+1}: FAILED ({result['reason']})")
    
    stats = channel.get_channel_statistics()
    
    print(f"\nResults:")
    print(f"  Success rate: {successes}/50 ({successes/50*100:.1f}%)")
    print(f"  Average latency: {np.mean(latencies):.2f} ms")
    print(f"  Latency std: {np.std(latencies):.2f} ms")
    print(f"  Actual loss rate: {stats['actual_loss_rate']*100:.2f}%")
    
    return stats


if __name__ == "__main__":
    demonstrate_network_simulator()
