"""
Training Script for Heart Disease Risk Assessment Model
Trains models on real ECG and clinical data instead of synthetic data.

Usage:
    python train_real_data.py
    
Before running:
    1. Put all ECG CSV files in training_data/ecg_files/
    2. Create training_data/clinical_data/clinical_data.csv with your data and labels
    3. See training_data/README_TRAINING_DATA.md for detailed instructions
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from main import (
    load_ecg, preprocess_ecg, detect_rpeaks, extract_ecg_features,
    preprocess_clinical, build_feature_vector
)


def load_training_data(project_root: Path):
    """Load clinical data CSV and verify paths."""
    training_data_dir = project_root / "training_data"
    clinical_data_path = training_data_dir / "clinical_data" / "clinical_data.csv"
    ecg_files_dir = training_data_dir / "ecg_files"
    
    # Check if files exist
    if not clinical_data_path.exists():
        raise FileNotFoundError(
            f"Clinical data file not found at {clinical_data_path}\n"
            f"Please create it following the template in training_data/README_TRAINING_DATA.md"
        )
    
    if not ecg_files_dir.exists():
        raise FileNotFoundError(
            f"ECG files directory not found at {ecg_files_dir}\n"
            f"Please create it and add your ECG CSV files"
        )
    
    # Load clinical data
    clinical_df = pd.read_csv(clinical_data_path)
    print(f"✅ Loaded clinical data: {len(clinical_df)} samples")
    print(f"Columns: {list(clinical_df.columns)}\n")
    
    # Check required columns
    required_cols = ['patient_id', 'ecg_filename', 'age', 'bp_systolic', 'bp_diastolic', 
                     'cholesterol', 'gender', 'bmi', 'smoking', 'heart_disease_label']
    missing_cols = [col for col in required_cols if col not in clinical_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("Clinical data preview:")
    print(clinical_df.head())
    print(f"\nLabel distribution:")
    print(clinical_df['heart_disease_label'].value_counts())
    print()
    
    return clinical_df, ecg_files_dir


def process_ecg_files(clinical_df: pd.DataFrame, ecg_files_dir: Path):
    """Process ECG files and extract features."""
    X = []
    y = []
    failed_samples = []
    
    print("Processing ECG files and extracting features...")
    print("This may take a few minutes depending on your dataset size.\n")
    
    for idx, row in tqdm(clinical_df.iterrows(), total=len(clinical_df), desc="Processing"):
        try:
            # Get ECG file path
            ecg_filename = row['ecg_filename']
            ecg_path = ecg_files_dir / ecg_filename
            
            if not ecg_path.exists():
                print(f"⚠️  Warning: ECG file not found: {ecg_filename} (skipping sample {row['patient_id']})")
                failed_samples.append(row['patient_id'])
                continue
            
            # Load and process ECG
            signal, fs = load_ecg(ecg_path)
            signal, fs = preprocess_ecg(signal, fs)
            peaks = detect_rpeaks(signal, fs)
            
            # Extract ECG features
            ecg_features = extract_ecg_features(peaks, signal, fs)
            
            # Prepare clinical features (ensure correct types)
            clinical_dict = {
                'age': float(row['age']),
                'bp_systolic': float(row['bp_systolic']),
                'bp_diastolic': float(row['bp_diastolic']),
                'cholesterol': float(row['cholesterol']),
                'gender': str(row['gender']),  # Keep as string, preprocess_clinical will convert
                'bmi': float(row['bmi']),
                'smoking': str(row['smoking'])  # Keep as string, preprocess_clinical will convert
            }
            
            # Preprocess clinical features (convert strings to numeric)
            clinical_features = preprocess_clinical(clinical_dict)
            
            # Build feature vector
            feature_vec = build_feature_vector(clinical_features, ecg_features)
            X.append(feature_vec.flatten())
            
            # Get label
            y.append(int(row['heart_disease_label']))
            
        except Exception as e:
            print(f"❌ Error processing {row['patient_id']} ({ecg_filename}): {e}")
            failed_samples.append(row['patient_id'])
            continue
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Successfully processed: {len(X)} samples")
    print(f"❌ Failed samples: {len(failed_samples)}")
    if failed_samples:
        print(f"   Failed patient IDs: {failed_samples}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"  Class 0 (No disease): {np.sum(y == 0)} samples")
    print(f"  Class 1 (Disease): {np.sum(y == 1)} samples\n")
    
    return X, y


def train_models(X_train, X_test, y_train, y_test):
    """Train Logistic Regression and Random Forest models."""
    results = {}
    
    # Train Logistic Regression
    print("="*60)
    print("Training Logistic Regression...")
    print("="*60)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    
    print("\nLogistic Regression Results:")
    print("-" * 60)
    print(classification_report(y_test, lr_pred))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, lr_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, lr_pred))
    
    results['lr'] = {
        'model': lr_model,
        'pred': lr_pred,
        'proba': lr_proba,
        'auc': roc_auc_score(y_test, lr_proba)
    }
    
    # Train Random Forest
    print("\n" + "="*60)
    print("Training Random Forest...")
    print("="*60)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    print("\nRandom Forest Results:")
    print("-" * 60)
    print(classification_report(y_test, rf_pred))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, rf_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))
    
    results['rf'] = {
        'model': rf_model,
        'pred': rf_pred,
        'proba': rf_proba,
        'auc': roc_auc_score(y_test, rf_proba)
    }
    
    return results


def save_models(models_dir: Path, lr_model, rf_model, scaler):
    """Save trained models and scaler to disk."""
    models_dir.mkdir(exist_ok=True)
    
    # Save Logistic Regression as baseline
    baseline_path = models_dir / "baseline_model.pkl"
    joblib.dump(lr_model, baseline_path)
    print(f"✅ Saved baseline model to {baseline_path}")
    
    # Save Random Forest
    rf_path = models_dir / "random_forest_model.pkl"
    joblib.dump(rf_model, rf_path)
    print(f"✅ Saved Random Forest model to {rf_path}")
    
    # Save scaler
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved scaler to {scaler_path}")
    
    print("\n🎉 Models saved successfully! You can now use them in the Streamlit app.")


def plot_roc_curves(y_test, lr_proba, rf_proba, save_path=None):
    """Plot ROC curves for model comparison."""
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, lr_proba):.3f})", linewidth=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, rf_proba):.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Model Comparison (Trained on Real Data)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curve plot to {save_path}")
    
    plt.show()


def main():
    """Main training pipeline."""
    print("="*60)
    print("Heart Disease Risk Assessment - Model Training")
    print("Training on Real Data")
    print("="*60)
    print()
    
    # Get project root
    project_root = Path(__file__).parent
    
    try:
        # Step 1: Load training data
        clinical_df, ecg_files_dir = load_training_data(project_root)
        
        # Step 2: Process ECG files and extract features
        X, y = process_ecg_files(clinical_df, ecg_files_dir)
        
        if len(X) == 0:
            raise ValueError("No samples were successfully processed. Please check your data.")
        
        # Step 3: Split data and scale features
        print("="*60)
        print("Splitting and scaling data...")
        print("="*60)
        
        # Split into train and test sets
        if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            print("⚠️  Warning: Not enough samples for stratified split. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"  Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"  Class 0: {np.sum(y_test == 0)}, Class 1: {np.sum(y_test == 1)}\n")
        
        # Step 4: Train models
        results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Step 5: Save models
        print("\n" + "="*60)
        print("Saving Models...")
        print("="*60)
        models_dir = project_root / "models"
        save_models(models_dir, results['lr']['model'], results['rf']['model'], scaler)
        
        # Step 6: Plot ROC curves
        print("\n" + "="*60)
        print("Generating ROC Curves...")
        print("="*60)
        plot_path = project_root / "models" / "roc_curves.png"
        plot_roc_curves(y_test, results['lr']['proba'], results['rf']['proba'], plot_path)
        
        print("\n" + "="*60)
        print("✅ Training complete! Models are ready for production use.")
        print("="*60)
        print("\nNext steps:")
        print("  1. Restart your Streamlit app: .\\start_streamlit.bat")
        print("  2. The app will now use models trained on your real data!")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. ECG files are in training_data/ecg_files/")
        print("  2. clinical_data.csv exists in training_data/clinical_data/")
        print("  3. See training_data/README_TRAINING_DATA.md for help")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


