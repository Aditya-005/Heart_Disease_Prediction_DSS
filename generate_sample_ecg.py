"""Generate synthetic ECG sample data"""
import numpy as np
from pathlib import Path

# Parameters
duration = 10  # seconds
fs = 500  # Hz
heart_rate = 72  # bpm
rr_interval = 60 / heart_rate  # seconds
n_samples = int(duration * fs)

# Time vector
t = np.linspace(0, duration, n_samples)

# Initialize ECG signal
ecg = np.zeros(n_samples)

# Generate QRS complexes at regular intervals
num_beats = int(duration / rr_interval)
for i in range(num_beats):
    peak_time = i * rr_interval
    idx = int(peak_time * fs)
    
    # QRS complex (simplified Gaussian-like shape)
    window_samples = int(0.2 * fs)  # 200ms window
    qrs_width = window_samples
    
    # Create QRS shape (Gaussian approximation)
    center = idx
    for j in range(-window_samples//2, window_samples//2):
        pos = center + j
        if 0 <= pos < n_samples:
            # Gaussian-like shape
            gauss_val = 2.0 * np.exp(-(j**2) / (2 * (qrs_width * 0.05)**2))
            ecg[pos] += gauss_val

# Add baseline wander (0.5 Hz)
baseline = 0.3 * np.sin(2 * np.pi * 0.5 * t)
ecg += baseline

# Add small amount of noise
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, 0.05, n_samples)
ecg += noise

# Save to CSV
output_path = Path(__file__).parent / "sample_data" / "sample_ecg.csv"
np.savetxt(output_path, ecg, delimiter=',', fmt='%.6f')
print(f"Sample ECG generated: {output_path} ({len(ecg)} samples, {fs} Hz)")

