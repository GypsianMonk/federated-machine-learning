"""
ECG Signal Generator and Dataset Manager for Federated Learning
Simulates realistic ECG data with arrhythmias distributed across hospitals (Non-IID)
"""

import numpy as np
from scipy.signal import resample
from typing import Tuple, Dict, List
import json
import os

class ECGSignalGenerator:
    """Generates synthetic ECG signals with various arrhythmia patterns"""
    
    def __init__(self, sampling_rate: int = 360):
        self.sampling_rate = sampling_rate
        self.beat_duration = 1.0  # seconds
        
    def generate_normal_beat(self, length: int = 360) -> np.ndarray:
        """Generate a normal sinus rhythm beat using Pan-Tompkins model"""
        t = np.linspace(0, self.beat_duration, length)
        
        # P wave (atrial depolarization)
        p_wave = 0.15 * np.exp(-((t - 0.2) ** 2) / 0.01)
        
        # QRS complex (ventricular depolarization)
        q_wave = -0.1 * np.exp(-((t - 0.35) ** 2) / 0.005)
        r_wave = 1.2 * np.exp(-((t - 0.37) ** 2) / 0.002)
        s_wave = -0.2 * np.exp(-((t - 0.39) ** 2) / 0.005)
        qrs = q_wave + r_wave + s_wave
        
        # T wave (ventricular repolarization)
        t_wave = 0.3 * np.exp(-((t - 0.55) ** 2) / 0.03)
        
        # U wave (optional, small)
        u_wave = 0.05 * np.exp(-((t - 0.7) ** 2) / 0.02)
        
        ecg = p_wave + qrs + t_wave + u_wave
        
        # Add baseline wander and noise
        baseline = 0.02 * np.sin(2 * np.pi * 0.5 * t)
        noise = 0.02 * np.random.randn(length)
        
        return ecg + baseline + noise
    
    def generate_pvc_beat(self, length: int = 360) -> np.ndarray:
        """Premature Ventricular Contraction - wide QRS, no P wave"""
        t = np.linspace(0, self.beat_duration, length)
        
        # No P wave
        # Wide QRS complex
        q_wave = -0.3 * np.exp(-((t - 0.35) ** 2) / 0.015)
        r_wave = 1.5 * np.exp(-((t - 0.38) ** 2) / 0.008)
        s_wave = -0.4 * np.exp(-((t - 0.42) ** 2) / 0.015)
        qrs = q_wave + r_wave + s_wave
        
        # Inverted T wave
        t_wave = -0.4 * np.exp(-((t - 0.6) ** 2) / 0.04)
        
        ecg = qrs + t_wave
        noise = 0.02 * np.random.randn(length)
        
        return ecg + noise
    
    def generate_apc_beat(self, length: int = 360) -> np.ndarray:
        """Atrial Premature Contraction - early P wave"""
        t = np.linspace(0, self.beat_duration, length)
        
        # Early, abnormal P wave
        p_wave = 0.2 * np.exp(-((t - 0.15) ** 2) / 0.008)
        
        # Normal QRS
        q_wave = -0.1 * np.exp(-((t - 0.3) ** 2) / 0.005)
        r_wave = 1.2 * np.exp(-((t - 0.32) ** 2) / 0.002)
        s_wave = -0.2 * np.exp(-((t - 0.34) ** 2) / 0.005)
        qrs = q_wave + r_wave + s_wave
        
        t_wave = 0.3 * np.exp(-((t - 0.5) ** 2) / 0.03)
        
        ecg = p_wave + qrs + t_wave
        noise = 0.02 * np.random.randn(length)
        
        return ecg + noise
    
    def generate_ecg_sequence(self, num_beats: int, arrhythmia_type: str = 'normal', 
                             arrhythmia_ratio: float = 0.1) -> np.ndarray:
        """Generate a sequence of ECG beats with specified arrhythmia ratio"""
        signal = []
        
        for i in range(num_beats):
            if arrhythmia_type == 'normal' or np.random.random() > arrhythmia_ratio:
                beat = self.generate_normal_beat()
            elif arrhythmia_type == 'pvc':
                beat = self.generate_pvc_beat()
            elif arrhythmia_type == 'apc':
                beat = self.generate_apc_beat()
            else:
                beat = self.generate_normal_beat()
            
            signal.append(beat)
        
        return np.concatenate(signal)


class ECGDatasetManager:
    """Manages ECG dataset distribution across hospitals (Non-IID simulation)"""
    
    def __init__(self, num_hospitals: int = 5, samples_per_hospital: int = 1000):
        self.num_hospitals = num_hospitals
        self.samples_per_hospital = samples_per_hospital
        self.generator = ECGSignalGenerator()
        self.hospital_data: Dict[int, Dict] = {}
        
    def create_heterogeneous_distribution(self) -> Dict[int, Dict]:
        """
        Create Non-IID data distribution simulating different hospital populations
        Each hospital has different arrhythmia prevalence
        """
        # Define hospital-specific arrhythmia profiles
        hospital_profiles = [
            {'name': 'Cardiac Center', 'pvc_ratio': 0.3, 'apc_ratio': 0.1},
            {'name': 'General Hospital', 'pvc_ratio': 0.1, 'apc_ratio': 0.05},
            {'name': 'Geriatric Clinic', 'pvc_ratio': 0.25, 'apc_ratio': 0.2},
            {'name': 'Pediatric Ward', 'pvc_ratio': 0.05, 'apc_ratio': 0.15},
            {'name': 'Emergency Dept', 'pvc_ratio': 0.35, 'apc_ratio': 0.25},
        ]
        
        for hospital_id in range(self.num_hospitals):
            profile = hospital_profiles[hospital_id % len(hospital_profiles)]
            
            # Generate data samples
            X_data = []
            y_labels = []
            
            for _ in range(self.samples_per_hospital):
                # Determine beat type based on hospital profile
                rand_val = np.random.random()
                
                if rand_val < profile['pvc_ratio']:
                    arrhythmia_type = 'pvc'
                    label = 1  # Abnormal
                elif rand_val < profile['pvc_ratio'] + profile['apc_ratio']:
                    arrhythmia_type = 'apc'
                    label = 1  # Abnormal
                else:
                    arrhythmia_type = 'normal'
                    label = 0  # Normal
                
                # Generate 2-beat sequence for context
                ecg_signal = self.generator.generate_ecg_sequence(
                    num_beats=2, 
                    arrhythmia_type=arrhythmia_type,
                    arrhythmia_ratio=0.8  # High ratio to ensure the type appears
                )
                
                X_data.append(ecg_signal[:720])  # Take first 2 beats (720 samples)
                y_labels.append(label)
            
            self.hospital_data[hospital_id] = {
                'name': profile['name'],
                'X': np.array(X_data),
                'y': np.array(y_labels),
                'pvc_ratio': profile['pvc_ratio'],
                'apc_ratio': profile['apc_ratio'],
                'abnormal_ratio': profile['pvc_ratio'] + profile['apc_ratio'],
                'size': len(X_data)
            }
        
        return self.hospital_data
    
    def get_train_test_split(self, hospital_id: int, test_ratio: float = 0.2) -> Tuple:
        """Get train/test split for a specific hospital"""
        if hospital_id not in self.hospital_data:
            raise ValueError(f"Hospital {hospital_id} not found")
        
        data = self.hospital_data[hospital_id]
        X, y = data['X'], data['y']
        
        # Shuffle indices
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_ratio)
        
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]
        
        return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'num_hospitals': self.num_hospitals,
            'samples_per_hospital': self.samples_per_hospital,
            'total_samples': self.num_hospitals * self.samples_per_hospital,
            'hospitals': {}
        }
        
        for hid, data in self.hospital_data.items():
            stats['hospitals'][hid] = {
                'name': data['name'],
                'size': data['size'],
                'abnormal_ratio': data['abnormal_ratio'],
                'pvc_ratio': data['pvc_ratio'],
                'apc_ratio': data['apc_ratio']
            }
        
        return stats
    
    def save_dataset(self, path: str = 'data/ecg_dataset.json'):
        """Save dataset statistics to JSON"""
        stats = self.get_statistics()
        
        # Save sample signals for visualization
        samples = {}
        for hid, data in self.hospital_data.items():
            samples[f'hospital_{hid}'] = {
                'sample_signal': data['X'][0].tolist(),
                'label': int(data['y'][0]),
                'name': data['name']
            }
        
        output = {
            'statistics': stats,
            'samples': samples
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return path


if __name__ == '__main__':
    # Test ECG dataset generation
    print("Generating ECG dataset for federated learning...")
    
    manager = ECGDatasetManager(num_hospitals=5, samples_per_hospital=500)
    hospital_data = manager.create_heterogeneous_distribution()
    
    # Print statistics
    stats = manager.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"Total Hospitals: {stats['num_hospitals']}")
    print(f"Total Samples: {stats['total_samples']}")
    print("\nHospital Distribution (Non-IID):")
    
    for hid, hstats in stats['hospitals'].items():
        print(f"  {hstats['name']}: {hstats['size']} samples, "
              f"{hstats['abnormal_ratio']*100:.1f}% abnormal")
    
    # Save dataset info
    saved_path = manager.save_dataset('results/ecg_dataset_info.json')
    print(f"\nDataset info saved to: {saved_path}")
    
    # Test train/test split
    train_data, test_data = manager.get_train_test_split(0)
    print(f"\nHospital 0 - Train: {len(train_data[0])}, Test: {len(test_data[0])}")
    print(f"Hospital 0 - Train labels distribution: "
          f"{np.sum(train_data[1])}/{len(train_data[1])} abnormal")
