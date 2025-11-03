"""
AI-Driven Predictive Healthcare System for Early Detection and Risk Assessment of Heart Disease
Core Processing Module

This module provides functions for ECG processing, feature extraction, and heart disease risk prediction.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import pickle
import warnings
warnings.filterwarnings('ignore')

# ECG processing
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    logging.warning("wfdb not available. ECG loading from .dat/.hea files will be limited.")

from scipy import signal as sp_signal
from scipy.signal import find_peaks, butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_SAMPLING_FREQ = 500  # Hz
MIN_ECG_LENGTH = 5  # seconds
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "baseline_model.pkl"
DEFAULT_SCALER_PATH = Path(__file__).parent / "models" / "scaler.pkl"


def load_ecg(record_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    """
    Load ECG signal from file (wfdb format or CSV).
    
    Args:
        record_path: Path to ECG record (without extension, for wfdb) or CSV file
        
    Returns:
        Tuple of (signal, sampling_frequency)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported
    """
    record_path = Path(record_path)
    
    # Try CSV first (simple format)
    csv_path = record_path if record_path.suffix == '.csv' else record_path.with_suffix('.csv')
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, header=None)
            signal = df.iloc[:, 0].values.astype(float)
            # Assume default sampling frequency for CSV
            fs = DEFAULT_SAMPLING_FREQ
            logger.info(f"Loaded ECG from CSV: {len(signal)} samples")
            return signal, fs
        except Exception as e:
            logger.warning(f"Failed to load as CSV: {e}")
    
    # Try wfdb format (.dat/.hea)
    if WFDB_AVAILABLE:
        try:
            # Remove extension if present
            base_path = record_path.with_suffix('')
            record = wfdb.rdrecord(str(base_path))
            
            # Use first channel (lead)
            signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
            fs = record.fs
            
            logger.info(f"Loaded ECG from wfdb: {len(signal)} samples at {fs} Hz")
            return signal, fs
        except Exception as e:
            logger.warning(f"Failed to load as wfdb: {e}")
    
    raise FileNotFoundError(f"Could not load ECG from {record_path}. Supported: .csv, .dat/.hea (wfdb)")


def preprocess_ecg(signal: np.ndarray, fs: float, target_fs: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Preprocess ECG signal: denoise, resample if needed.
    
    Args:
        signal: Raw ECG signal
        fs: Current sampling frequency
        target_fs: Target sampling frequency (default: keep original)
        
    Returns:
        Tuple of (processed_signal, sampling_frequency)
    """
    # Remove DC offset
    signal = signal - np.mean(signal)
    
    # Bandpass filter (0.5-40 Hz) to remove noise and baseline wander
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 40 / nyquist
    
    try:
        b, a = butter(4, [low, high], btype='band')
        signal = filtfilt(b, a, signal)
    except ValueError:
        # If filter fails, just use high-pass
        b, a = butter(4, low, btype='high')
        signal = filtfilt(b, a, signal)
    
    # Resample if target_fs is specified and different
    if target_fs and target_fs != fs:
        num_samples = int(len(signal) * target_fs / fs)
        signal = resample(signal, num_samples)
        fs = target_fs
        logger.info(f"Resampled to {fs} Hz")
    
    return signal, fs


def detect_rpeaks(signal: np.ndarray, fs: float, method: str = 'auto') -> np.ndarray:
    """
    Detect R-peaks in ECG signal using Pan-Tompkins-like algorithm or scipy fallback.
    
    Args:
        signal: Preprocessed ECG signal
        fs: Sampling frequency
        method: 'pan_tompkins', 'scipy', or 'auto'
        
    Returns:
        Array of R-peak indices
    """
    if method == 'auto':
        # Try Pan-Tompkins first, fallback to scipy
        try:
            peaks = _pan_tompkins_detection(signal, fs)
            if len(peaks) < 2:  # Too few peaks, use scipy fallback
                logger.warning("Pan-Tompkins found too few peaks, using scipy fallback")
                peaks = _scipy_peak_detection(signal, fs)
        except Exception as e:
            logger.warning(f"Pan-Tompkins failed: {e}, using scipy fallback")
            peaks = _scipy_peak_detection(signal, fs)
    elif method == 'pan_tompkins':
        peaks = _pan_tompkins_detection(signal, fs)
    else:  # scipy
        peaks = _scipy_peak_detection(signal, fs)
    
    logger.info(f"Detected {len(peaks)} R-peaks")
    return peaks


def _pan_tompkins_detection(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Simplified Pan-Tompkins R-peak detection.
    """
    # Differentiate
    diff_signal = np.diff(signal)
    
    # Square
    squared = diff_signal ** 2
    
    # Moving window integration (150ms window)
    window_size = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # Find peaks with adaptive threshold
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, properties = find_peaks(integrated, height=threshold, distance=int(0.4 * fs))
    
    # Map back to original signal (account for diff offset)
    peaks = peaks + 1
    
    # Refine: find actual max in signal near each detected peak
    refined_peaks = []
    search_window = int(0.1 * fs)
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(signal), peak + search_window)
        refined_peak = start + np.argmax(np.abs(signal[start:end]))
        refined_peaks.append(refined_peak)
    
    return np.array(refined_peaks)


def _scipy_peak_detection(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Simple scipy-based peak detection fallback.
    """
    # Estimate typical RR interval (0.6-1.2 seconds for normal heart rate)
    min_distance = int(0.6 * fs)
    max_distance = int(1.2 * fs)
    
    # Adaptive threshold
    threshold = np.mean(signal) + 2 * np.std(signal)
    
    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)
    
    return peaks


def extract_ecg_features(peaks: np.ndarray, signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract ECG features from R-peaks and signal.
    
    Args:
        peaks: R-peak indices
        signal: ECG signal
        fs: Sampling frequency
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    if len(peaks) < 2:
        logger.warning("Too few peaks for feature extraction, using defaults")
        features = {
            'rr_mean': 0.8,  # Default RR interval
            'rr_std': 0.1,
            'hrv_sdnn': 50.0,
            'hrv_rmssd': 30.0,
            'heart_rate': 75.0,
            'signal_quality': 0.5,
            'qrs_width': 0.1,
            'signal_variance': float(np.var(signal))
        }
        return features
    
    # RR intervals (in seconds)
    rr_intervals = np.diff(peaks) / fs
    
    # Basic RR statistics
    features['rr_mean'] = float(np.mean(rr_intervals))
    features['rr_std'] = float(np.std(rr_intervals))
    
    # Heart Rate Variability (HRV) metrics
    features['hrv_sdnn'] = float(np.std(rr_intervals) * 1000)  # SDNN in ms
    features['hrv_rmssd'] = float(np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000) if len(rr_intervals) > 1 else 0.0
    
    # Heart rate
    features['heart_rate'] = float(60.0 / features['rr_mean']) if features['rr_mean'] > 0 else 0.0
    
    # Signal quality: variance-based metric
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    features['signal_quality'] = float(1.0 / (1.0 + np.std(np.diff(signal_norm))))
    
    # QRS width estimation (simplified)
    qrs_widths = []
    for peak in peaks[:min(10, len(peaks))]:  # Sample first 10 peaks
        start = max(0, peak - int(0.05 * fs))
        end = min(len(signal), peak + int(0.05 * fs))
        segment = signal[start:end]
        threshold = np.max(segment) * 0.5
        above_threshold = np.where(np.abs(segment) > threshold)[0]
        if len(above_threshold) > 0:
            qrs_widths.append((above_threshold[-1] - above_threshold[0]) / fs)
    
    features['qrs_width'] = float(np.mean(qrs_widths)) if qrs_widths else 0.1
    
    # Signal variance
    features['signal_variance'] = float(np.var(signal))
    
    logger.info(f"Extracted {len(features)} ECG features")
    return features


def preprocess_clinical(clinical_dict: Dict) -> Dict[str, float]:
    """
    Validate and normalize clinical features.
    
    Args:
        clinical_dict: Dictionary with clinical parameters
        
    Returns:
        Validated and normalized clinical dictionary
    """
    validated = {}
    
    # Age (0-120)
    age = float(clinical_dict.get('age', 50))
    validated['age'] = max(0, min(120, age))
    
    # Blood pressure (systolic/diastolic in mmHg)
    bp_systolic = float(clinical_dict.get('bp_systolic', clinical_dict.get('systolic_bp', 120)))
    bp_diastolic = float(clinical_dict.get('bp_diastolic', clinical_dict.get('diastolic_bp', 80)))
    validated['bp_systolic'] = max(70, min(250, bp_systolic))
    validated['bp_diastolic'] = max(40, min(150, bp_diastolic))
    
    # Cholesterol (mg/dL)
    cholesterol = float(clinical_dict.get('cholesterol', clinical_dict.get('chol', 200)))
    validated['cholesterol'] = max(100, min(400, cholesterol))
    
    # Gender (0=female, 1=male, 0.5=unknown)
    gender_str = str(clinical_dict.get('gender', 'unknown')).lower()
    if gender_str in ['m', 'male', '1']:
        validated['gender'] = 1.0
    elif gender_str in ['f', 'female', '0']:
        validated['gender'] = 0.0
    else:
        validated['gender'] = 0.5
    
    # Optional: BMI if provided
    if 'bmi' in clinical_dict:
        validated['bmi'] = float(clinical_dict['bmi'])
    else:
        # Estimate from height/weight if available
        if 'height' in clinical_dict and 'weight' in clinical_dict:
            height_m = float(clinical_dict['height']) / 100  # cm to m
            weight_kg = float(clinical_dict['weight'])
            validated['bmi'] = weight_kg / (height_m ** 2) if height_m > 0 else 25.0
        else:
            validated['bmi'] = 25.0  # Default
    
    # Smoking status (0=no, 1=yes, 0.5=unknown)
    smoking = str(clinical_dict.get('smoking', 'unknown')).lower()
    if smoking in ['yes', 'y', '1', 'smoker']:
        validated['smoking'] = 1.0
    elif smoking in ['no', 'n', '0', 'non-smoker']:
        validated['smoking'] = 0.0
    else:
        validated['smoking'] = 0.5
    
    logger.info(f"Preprocessed {len(validated)} clinical features")
    return validated


def build_feature_vector(clinical_features: Dict[str, float], ecg_features: Dict[str, float]) -> np.ndarray:
    """
    Combine clinical and ECG features into a single feature vector.
    
    Args:
        clinical_features: Clinical feature dictionary
        ecg_features: ECG feature dictionary
        
    Returns:
        Combined feature vector as numpy array
    """
    # Define feature order (must match training)
    feature_names = [
        'age',
        'bp_systolic',
        'bp_diastolic',
        'cholesterol',
        'gender',
        'bmi',
        'smoking',
        'rr_mean',
        'rr_std',
        'hrv_sdnn',
        'hrv_rmssd',
        'heart_rate',
        'signal_quality',
        'qrs_width',
        'signal_variance'
    ]
    
    feature_vector = []
    for name in feature_names:
        if name in clinical_features:
            feature_vector.append(clinical_features[name])
        elif name in ecg_features:
            feature_vector.append(ecg_features[name])
        else:
            # Default value
            defaults = {
                'age': 50.0,
                'bp_systolic': 120.0,
                'bp_diastolic': 80.0,
                'cholesterol': 200.0,
                'gender': 0.5,
                'bmi': 25.0,
                'smoking': 0.5,
                'rr_mean': 0.8,
                'rr_std': 0.1,
                'hrv_sdnn': 50.0,
                'hrv_rmssd': 30.0,
                'heart_rate': 75.0,
                'signal_quality': 0.5,
                'qrs_width': 0.1,
                'signal_variance': 1.0
            }
            feature_vector.append(defaults.get(name, 0.0))
    
    return np.array(feature_vector).reshape(1, -1)


def load_model(model_path: Optional[Union[str, Path]] = None) -> Tuple:
    """
    Load trained model and scaler.
    
    Args:
        model_path: Path to model file (default: DEFAULT_MODEL_PATH)
        
    Returns:
        Tuple of (model, scaler)
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train a model first.")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
    # Try to load scaler
    scaler_path = model_path.parent / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        logger.warning("Scaler not found, will use default StandardScaler")
        scaler = None
    
    return model, scaler


def predict(features: np.ndarray, model=None, scaler=None, model_path: Optional[Union[str, Path]] = None) -> Tuple[float, int]:
    """
    Predict heart disease risk probability.
    
    Args:
        features: Feature vector (1, n_features)
        model: Pre-loaded model (optional)
        scaler: Pre-loaded scaler (optional)
        model_path: Path to model file (if model not provided)
        
    Returns:
        Tuple of (probability, predicted_label)
    """
    if model is None:
        model, scaler = load_model(model_path)
    
    # Scale features if scaler available
    if scaler is not None:
        features = scaler.transform(features)
    
    # Predict
    try:
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)[0, 1]  # Probability of positive class
        else:
            # If model doesn't have predict_proba, use decision function
            prob = 1.0 / (1.0 + np.exp(-model.decision_function(features)[0]))
        
        label = 1 if prob >= 0.5 else 0
        
        logger.info(f"Prediction: probability={prob:.4f}, label={label}")
        return float(prob), int(label)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def decision_logic(prob: float) -> Tuple[str, str]:
    """
    Convert prediction probability into risk category and recommendation.
    
    Args:
        prob: Probability of heart disease (0-1)
        
    Returns:
        Tuple of (risk_category, recommendation_text)
    """
    if prob < 0.3:
        category = "Low"
        recommendation = "Low risk detected. Maintain healthy lifestyle with regular exercise and balanced diet. Annual checkups recommended."
    elif prob < 0.6:
        category = "Medium"
        recommendation = "Moderate risk detected. Consult with a cardiologist for further evaluation. Consider lifestyle modifications and regular monitoring."
    else:
        category = "High"
        recommendation = "High risk detected. Immediate consultation with a cardiologist is strongly recommended. Follow-up tests may be necessary."
    
    return category, recommendation


def save_prediction(record: Dict, output_path: Optional[Union[str, Path]] = None) -> None:
    """
    Save prediction record to CSV.
    
    Args:
        record: Dictionary with prediction data
        output_path: Path to save CSV (default: predictions.csv in project root)
    """
    if output_path is None:
        output_path = Path(__file__).parent / "predictions.csv"
    
    output_path = Path(output_path)
    
    # Check if file exists
    if output_path.exists():
        df = pd.read_csv(output_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved prediction to {output_path}")


def process_full_pipeline(ecg_path: Union[str, Path], clinical_dict: Dict, 
                          model_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Complete pipeline: load ECG, extract features, predict, and return results.
    
    Args:
        ecg_path: Path to ECG file
        clinical_dict: Clinical parameters dictionary
        model_path: Optional path to model
        
    Returns:
        Dictionary with all results
    """
    try:
        # Load ECG
        signal, fs = load_ecg(ecg_path)
        
        # Preprocess
        signal, fs = preprocess_ecg(signal, fs)
        
        # Detect R-peaks
        peaks = detect_rpeaks(signal, fs)
        
        # Extract ECG features
        ecg_features = extract_ecg_features(peaks, signal, fs)
        
        # Preprocess clinical data
        clinical_features = preprocess_clinical(clinical_dict)
        
        # Build feature vector
        feature_vector = build_feature_vector(clinical_features, ecg_features)
        
        # Load model and predict
        model, scaler = load_model(model_path)
        prob, label = predict(feature_vector, model=model, scaler=scaler)
        
        # Decision logic
        risk_category, recommendation = decision_logic(prob)
        
        # Compile results
        results = {
            'probability': prob,
            'predicted_label': label,
            'risk_category': risk_category,
            'recommendation': recommendation,
            'clinical_features': clinical_features,
            'ecg_features': ecg_features,
            'num_rpeaks': len(peaks),
            'signal_length': len(signal),
            'sampling_freq': fs
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    """
    CLI interface for testing single sample.
    Usage: python main.py --ecg sample_data/sample_ecg.csv --age 55 --bp_systolic 140 --bp_diastolic 90 --cholesterol 220
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Heart Disease Risk Prediction CLI')
    parser.add_argument('--ecg', type=str, required=True, help='Path to ECG file')
    parser.add_argument('--age', type=int, required=True, help='Age')
    parser.add_argument('--bp_systolic', type=int, required=True, help='Systolic BP')
    parser.add_argument('--bp_diastolic', type=int, required=True, help='Diastolic BP')
    parser.add_argument('--cholesterol', type=int, required=True, help='Cholesterol')
    parser.add_argument('--gender', type=str, default='unknown', help='Gender (M/F)')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    parser.add_argument('--save', action='store_true', help='Save prediction to CSV')
    
    args = parser.parse_args()
    
    clinical_dict = {
        'age': args.age,
        'bp_systolic': args.bp_systolic,
        'bp_diastolic': args.bp_diastolic,
        'cholesterol': args.cholesterol,
        'gender': args.gender
    }
    
    try:
        results = process_full_pipeline(args.ecg, clinical_dict, args.model)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Risk Probability: {results['probability']:.4f}")
        print(f"Risk Category: {results['risk_category']}")
        print(f"\nRecommendation:")
        print(results['recommendation'])
        print(f"\nECG Features:")
        for key, value in results['ecg_features'].items():
            print(f"  {key}: {value:.4f}")
        print("="*50)
        
        if args.save:
            save_prediction({
                'ecg_path': str(args.ecg),
                'age': args.age,
                'bp_systolic': args.bp_systolic,
                'bp_diastolic': args.bp_diastolic,
                'cholesterol': args.cholesterol,
                'probability': results['probability'],
                'risk_category': results['risk_category']
            })
            print("\nPrediction saved to predictions.csv")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

