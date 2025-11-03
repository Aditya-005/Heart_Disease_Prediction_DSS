"""
Generate Synthetic Training Dataset
Creates realistic synthetic ECG signals and clinical data for training

Usage:
    python generate_training_dataset.py --num-samples 500 --output training_data/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from main import extract_ecg_features, detect_rpeaks, preprocess_ecg


def generate_realistic_ecg(duration_seconds=10, fs=500, heart_rate=72, noise_level=0.05, 
                          has_abnormality=False):
    """
    Generate a realistic synthetic ECG signal.
    
    Args:
        duration_seconds: Length of signal in seconds
        fs: Sampling frequency
        heart_rate: Heart rate in BPM
        noise_level: Amount of noise (0-1)
        has_abnormality: If True, add abnormalities (arrhythmia-like patterns)
    
    Returns:
        ECG signal as numpy array
    """
    t = np.arange(0, duration_seconds, 1/fs)
    ecg_signal = np.zeros_like(t)
    
    # Calculate RR interval from heart rate
    rr_interval = 60.0 / heart_rate  # in seconds
    num_beats = int(duration_seconds / rr_interval)
    
    # Generate ECG beat template (simplified)
    beat_duration = 0.8  # seconds
    beat_samples = int(beat_duration * fs)
    beat_template = np.zeros(beat_samples)
    
    # Create realistic QRS complex
    qrs_start = int(0.2 * beat_samples)
    qrs_end = int(0.3 * beat_samples)
    beat_template[qrs_start:qrs_end] = np.sin(np.linspace(0, 3*np.pi, qrs_end - qrs_start)) * 1.5
    
    # Add P wave
    p_start = int(0.05 * beat_samples)
    p_end = int(0.15 * beat_samples)
    beat_template[p_start:p_end] = np.sin(np.linspace(0, np.pi, p_end - p_start)) * 0.3
    
    # Add T wave
    t_start = int(0.35 * beat_samples)
    t_end = int(0.7 * beat_samples)
    t_wave = np.exp(-np.linspace(0, 5, t_end - t_start)) * np.sin(np.linspace(0, np.pi, t_end - t_start))
    beat_template[t_start:t_end] = t_wave * 0.5
    
    # Place beats in signal
    beat_spacing = int(rr_interval * fs)
    
    for i in range(num_beats + 1):  # +1 for safety
        position = i * beat_spacing
        
        if position + beat_samples <= len(ecg_signal):
            if has_abnormality and i % 3 == 0:  # Irregular beat every 3rd beat
                # Make beat wider and irregular
                abnormal_template = beat_template.copy()
                abnormal_template = np.interp(
                    np.linspace(0, len(abnormal_template), int(len(abnormal_template) * 1.3)),
                    np.arange(len(abnormal_template)),
                    abnormal_template
                )[:len(beat_template)]
                ecg_signal[position:position+len(abnormal_template)] += abnormal_template * 0.7
            else:
                ecg_signal[position:position+len(beat_template)] += beat_template
        
        # Add slight variability to heart rate
        beat_spacing = int((rr_interval + np.random.normal(0, 0.02)) * fs)
    
    # Add noise (baseline wander, muscle artifacts, powerline)
    noise = (
        np.random.normal(0, noise_level, len(ecg_signal)) +  # White noise
        0.3 * noise_level * np.sin(2 * np.pi * 0.5 * t) +  # Baseline wander (0.5 Hz)
        0.2 * noise_level * np.sin(2 * np.pi * 50 * t) +    # Powerline interference (50 Hz)
        0.1 * noise_level * np.random.randn(len(ecg_signal)) * np.sin(2 * np.pi * 60 * t)  # Muscle artifact
    )
    
    ecg_signal += noise
    
    # Normalize
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-10)
    
    return ecg_signal


def generate_clinical_data(num_samples=500, seed=42):
    """Generate realistic synthetic clinical data."""
    np.random.seed(seed)
    
    data = []
    
    for i in range(num_samples):
        # Simulate patient demographics
        age = np.random.randint(30, 80)
        
        # Blood pressure (normal: 120/80, high: >140/90)
        bp_systolic = np.random.normal(130, 20)
        bp_systolic = np.clip(bp_systolic, 90, 180)
        bp_diastolic = np.random.normal(85, 15)
        bp_diastolic = np.clip(bp_diastolic, 60, 120)
        
        # Cholesterol (normal: <200, high: >240)
        cholesterol = np.random.normal(220, 40)
        cholesterol = np.clip(cholesterol, 150, 300)
        
        # Gender
        gender = np.random.choice(['M', 'F'], p=[0.55, 0.45])
        
        # BMI (normal: 18.5-25, overweight: 25-30, obese: >30)
        bmi = np.random.normal(26, 5)
        bmi = np.clip(bmi, 18, 40)
        
        # Smoking (realistic distribution)
        smoking = np.random.choice(['yes', 'no'], p=[0.25, 0.75])
        
        # Determine heart disease risk based on clinical factors
        risk_score = (
            (age - 50) / 30 * 0.3 +  # Age risk
            (bp_systolic - 120) / 60 * 0.2 +  # BP risk
            (cholesterol - 200) / 100 * 0.2 +  # Cholesterol risk
            (bmi - 25) / 15 * 0.15 +  # BMI risk
            (1 if smoking == 'yes' else 0) * 0.15 +  # Smoking risk
            np.random.normal(0, 0.1)  # Random noise
        )
        
        # Convert to probability
        risk_prob = 1 / (1 + np.exp(-risk_score))  # Sigmoid
        
        # Generate label (1 = disease, 0 = no disease)
        # Higher risk score = more likely to have disease
        heart_disease = 1 if risk_prob > 0.5 else 0
        
        # Adjust based on clinical thresholds
        if age > 65 and bp_systolic > 140:
            heart_disease = 1
        if cholesterol > 240 and smoking == 'yes':
            heart_disease = 1
        if age < 40 and bp_systolic < 120 and cholesterol < 180:
            heart_disease = 0
        
        data.append({
            'patient_id': f'patient_{i+1:03d}',
            'ecg_filename': f'ecg_{i+1:03d}.csv',
            'age': int(age),
            'bp_systolic': int(bp_systolic),
            'bp_diastolic': int(bp_diastolic),
            'cholesterol': int(cholesterol),
            'gender': gender,
            'bmi': round(bmi, 1),
            'smoking': smoking,
            'heart_disease_label': heart_disease
        })
    
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training dataset')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='training_data',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    ecg_dir = output_dir / 'ecg_files'
    clinical_dir = output_dir / 'clinical_data'
    
    ecg_dir.mkdir(parents=True, exist_ok=True)
    clinical_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating Synthetic Training Dataset")
    print("="*60)
    print(f"Generating {args.num_samples} samples...")
    print()
    
    # Generate clinical data
    print("Generating clinical data...")
    clinical_df = generate_clinical_data(args.num_samples, args.seed)
    
    # Generate ECG files
    print("Generating ECG signals...")
    from tqdm import tqdm
    
    for idx, row in tqdm(clinical_df.iterrows(), total=len(clinical_df), desc="ECG files"):
        # Determine heart rate and abnormalities based on clinical data
        base_hr = np.random.normal(72, 10)
        
        if row['heart_disease_label'] == 1:
            # Patients with disease: irregular heart rate, higher variability
            heart_rate = base_hr + np.random.normal(0, 15)
            has_abnormality = np.random.choice([True, False], p=[0.4, 0.6])
            noise_level = 0.08  # More noise
        else:
            # Healthy patients: regular heart rate
            heart_rate = base_hr + np.random.normal(0, 5)
            has_abnormality = False
            noise_level = 0.05
        
        heart_rate = np.clip(heart_rate, 50, 110)
        
        # Generate ECG
        ecg_signal = generate_realistic_ecg(
            duration_seconds=10,
            fs=500,
            heart_rate=heart_rate,
            noise_level=noise_level,
            has_abnormality=has_abnormality
        )
        
        # Save ECG file
        ecg_path = ecg_dir / row['ecg_filename']
        np.savetxt(ecg_path, ecg_signal, delimiter=',')
    
    # Save clinical data
    clinical_path = clinical_dir / 'clinical_data.csv'
    clinical_df.to_csv(clinical_path, index=False)
    
    print()
    print("="*60)
    print("✅ Dataset Generation Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - ECG files: {len(list(ecg_dir.glob('*.csv')))} files in {ecg_dir}")
    print(f"  - Clinical data: {clinical_path}")
    print(f"\nLabel distribution:")
    print(clinical_df['heart_disease_label'].value_counts())
    print(f"\nNext step: Train your model!")
    print(f"  python train_real_data.py")
    print(f"  or")
    print(f"  .\\train_model.bat")
    print()


if __name__ == "__main__":
    main()

