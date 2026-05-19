"""
Drift Detector for Federated Learning
Detects concept drift in ECG data distributions using ADWIN and KS-test
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque
from scipy import stats


class ADWINDriftDetector:
    """
    Adaptive Windowing (ADWIN) algorithm for concept drift detection.
    Automatically adjusts window size based on rate of change.
    """
    
    def __init__(self, delta: float = 0.002, min_window: int = 10):
        self.delta = delta  # Confidence parameter
        self.min_window = min_window
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.drift_detected_count = 0
        
    def add_element(self, value: float) -> bool:
        """
        Add a new element and check for drift.
        
        Returns:
            True if drift detected, False otherwise
        """
        self.window.append(value)
        self.width += 1
        self.total += value
        
        # Update variance estimate
        if self.width > 1:
            self.variance = ((self.width - 1) * self.variance + 
                           (value - self.total/self.width)**2) / self.width
        
        # Check for drift when window is large enough
        if self.width >= self.min_window * 2:
            return self._check_drift()
        
        return False
    
    def _check_drift(self) -> bool:
        """Check for drift by comparing subwindows"""
        max_subwindow = self.width // 2
        
        for split_point in range(self.min_window, max_subwindow):
            # Calculate means of two subwindows
            window_list = list(self.window)
            left_mean = np.mean(window_list[:split_point])
            right_mean = np.mean(window_list[split_point:])
            
            # Hoeffding bound
            n1, n2 = split_point, self.width - split_point
            m = 1.0 / (1.0/n1 + 1.0/n2)
            epsilon = np.sqrt((1.0/(2*m)) * np.log(4.0/self.delta))
            
            # Check if difference exceeds bound
            if abs(left_mean - right_mean) > epsilon:
                # Drift detected - shrink window
                for _ in range(split_point):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.width -= 1
                
                self.drift_detected_count += 1
                return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Return detector statistics"""
        return {
            'window_size': self.width,
            'total_elements': len(self.window),
            'drift_events': self.drift_detected_count,
            'current_mean': self.total / max(self.width, 1),
            'variance': self.variance
        }
    
    def reset(self):
        """Reset the detector"""
        self.window.clear()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0


class KSDriftDetector:
    """
    Kolmogorov-Smirnov test based drift detector.
    Compares recent distribution to reference distribution.
    """
    
    def __init__(self, reference_data: np.ndarray = None, 
                 alpha: float = 0.01,
                 window_size: int = 50):
        self.reference_data = reference_data
        self.alpha = alpha
        self.window_size = window_size
        self.recent_window = deque(maxlen=window_size)
        self.drift_detected_count = 0
        self.tests_run = 0
        
    def set_reference(self, data: np.ndarray):
        """Set reference distribution"""
        self.reference_data = data
    
    def add_element(self, value: float) -> bool:
        """Add element and test for drift"""
        self.recent_window.append(value)
        
        if len(self.recent_window) < self.window_size or self.reference_data is None:
            return False
        
        self.tests_run += 1
        
        # Perform KS test
        statistic, p_value = stats.kstest(
            list(self.recent_window), 
            lambda x: np.searchsorted(np.sort(self.reference_data), x) / len(self.reference_data)
        )
        
        if p_value < self.alpha:
            self.drift_detected_count += 1
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Return detector statistics"""
        return {
            'window_size': len(self.recent_window),
            'tests_run': self.tests_run,
            'drift_events': self.drift_detected_count,
            'alpha': self.alpha,
            'has_reference': self.reference_data is not None
        }


def demonstrate_drift_detection():
    """Demonstrate drift detection on synthetic ECG features"""
    print("="*60)
    print("DRIFT DETECTION DEMONSTRATION")
    print("="*60)
    
    # Generate baseline ECG heart rate data (normal: 60-100 bpm)
    np.random.seed(42)
    baseline_hr = np.random.normal(75, 10, 200)  # Normal heart rate
    
    # Create detector with reference distribution
    ks_detector = KSDriftDetector(reference_data=baseline_hr, alpha=0.01)
    adwin_detector = ADWINDriftDetector(delta=0.002)
    
    print("\nSimulating patient heart rate monitoring...")
    print("Baseline: Normal sinus rhythm (75±10 bpm)")
    print("Event: Tachycardia onset at sample 150\n")
    
    drift_events_ks = []
    drift_events_adwin = []
    
    # Simulate 300 time steps
    for t in range(300):
        # Normal until t=150, then tachycardia (140±15 bpm)
        if t < 150:
            value = np.random.normal(75, 10)
        else:
            value = np.random.normal(140, 15)
        
        # Test for drift
        ks_drift = ks_detector.add_element(value)
        adwin_drift = adwin_detector.add_element(value)
        
        if ks_drift:
            drift_events_ks.append(t)
            print(f"  [KS-Test] Drift detected at t={t}!")
        
        if adwin_drift:
            drift_events_adwin.append(t)
            print(f"  [ADWIN] Drift detected at t={t}!")
    
    print(f"\nResults:")
    print(f"  KS-Test: {len(drift_events_ks)} drift events detected")
    print(f"  ADWIN: {len(drift_events_adwin)} drift events detected")
    
    ks_stats = ks_detector.get_statistics()
    adwin_stats = adwin_detector.get_statistics()
    
    print(f"\nKS-Test Statistics:")
    print(f"  Tests run: {ks_stats['tests_run']}")
    print(f"  Detection rate: {len(drift_events_ks)/max(ks_stats['tests_run'],1)*100:.1f}%")
    
    print(f"\nADWIN Statistics:")
    print(f"  Final window size: {adwin_stats['window_size']}")
    print(f"  Drift events: {adwin_stats['drift_events']}")
    
    return {
        'ks_detector': ks_stats,
        'adwin_detector': adwin_stats,
        'ks_events': drift_events_ks,
        'adwin_events': drift_events_adwin
    }


if __name__ == "__main__":
    demonstrate_drift_detection()
