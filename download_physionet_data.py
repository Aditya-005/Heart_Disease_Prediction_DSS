"""
Download ECG data from PhysioNet databases
Supports PTB Diagnostic Database and other PhysioNet datasets

Usage:
    python download_physionet_data.py --dataset ptbdb --output training_data/ecg_files/
    python download_physionet_data.py --dataset mitdb --output training_data/ecg_files/
"""

import argparse
import sys
from pathlib import Path
import requests
import os
from tqdm import tqdm
import zipfile
import shutil

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("Warning: wfdb package not available. Installing...")
    print("Please run: pip install wfdb")
    sys.exit(1)


def download_file(url: str, output_path: Path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_ptbdb(output_dir: Path, limit=None):
    """
    Download PTB Diagnostic Database from PhysioNet.
    
    Note: PhysioNet requires user authentication for some datasets.
    You may need to:
    1. Create an account at https://physionet.org/
    2. Sign the data use agreement
    3. Use wfdb to download directly (recommended method below)
    """
    print("="*60)
    print("Downloading PTB Diagnostic Database")
    print("="*60)
    print("\nNote: PhysioNet requires authentication for PTB-DB.")
    print("Please use wfdb library method (recommended) or")
    print("download manually from: https://physionet.org/content/ptbdb/")
    print()
    
    # Recommended: Use wfdb to download
    try:
        print("Using wfdb library to download records...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of records
        print("Fetching record list...")
        records = wfdb.get_record_list('ptbdb')
        
        if limit:
            records = records[:limit]
        
        print(f"Downloading {len(records)} records...")
        
        for record_name in tqdm(records, desc="Downloading"):
            try:
                # Download record
                record_path = output_dir / record_name
                wfdb.dl_database('ptbdb', [record_name], str(output_dir))
                
                # Convert to CSV for easier use
                record = wfdb.rdrecord(str(record_path))
                signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
                
                csv_path = record_path.with_suffix('.csv')
                import numpy as np
                np.savetxt(csv_path, signal, delimiter=',')
                
            except Exception as e:
                print(f"\nWarning: Failed to download {record_name}: {e}")
                continue
        
        print(f"\n✅ Downloaded records to {output_dir}")
        print(f"   Each record saved as .csv file")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAlternative: Download manually from:")
        print("https://physionet.org/content/ptbdb/")
        return False
    
    return True


def download_sample_records(output_dir: Path, num_records=10):
    """Download a few sample records for testing."""
    print("="*60)
    print("Downloading Sample Records")
    print("="*60)
    print("\nDownloading a few sample records for testing...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample record names from PTB-DB (patient001, patient002, etc.)
    sample_records = [f"patient{i:03d}" for i in range(1, min(num_records + 1, 100))]
    
    downloaded = 0
    for record_name in tqdm(sample_records, desc="Downloading"):
        try:
            record_path = output_dir / record_name
            wfdb.dl_database('ptbdb', [record_name], str(output_dir))
            
            # Convert to CSV
            record = wfdb.rdrecord(str(record_path))
            signal = record.p_signal[:, 0] if record.p_signal.ndim > 1 else record.p_signal
            
            csv_path = record_path.with_suffix('.csv')
            import numpy as np
            np.savetxt(csv_path, signal, delimiter=',')
            downloaded += 1
            
        except Exception as e:
            continue
    
    print(f"\n✅ Downloaded {downloaded} sample records to {output_dir}")
    return downloaded > 0


def main():
    parser = argparse.ArgumentParser(description='Download ECG data from PhysioNet')
    parser.add_argument('--dataset', type=str, choices=['ptbdb', 'mitdb', 'sample'],
                       default='sample', help='Dataset to download')
    parser.add_argument('--output', type=str, default='training_data/ecg_files',
                       help='Output directory for ECG files')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records to download')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of sample records (for sample mode)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if not WFDB_AVAILABLE:
        print("❌ wfdb package is required. Please install it:")
        print("   pip install wfdb")
        sys.exit(1)
    
    print("\n⚠️  Important: PhysioNet requires:")
    print("   1. Free account at https://physionet.org/")
    print("   2. Sign data use agreement for PTB-DB dataset")
    print("   3. Credentials may be required\n")
    
    if args.dataset == 'sample':
        success = download_sample_records(output_dir, args.num_samples)
    elif args.dataset == 'ptbdb':
        success = download_ptbdb(output_dir, args.limit)
    else:
        print(f"Dataset '{args.dataset}' download not yet implemented.")
        print("Use --dataset sample for testing, or --dataset ptbdb for full PTB-DB")
        success = False
    
    if success:
        print("\n✅ Download complete!")
        print(f"\nNext steps:")
        print(f"  1. Check downloaded files in: {output_dir}")
        print(f"  2. Create clinical_data.csv with labels")
        print(f"  3. Run: python train_real_data.py")
    else:
        print("\n❌ Download failed. See errors above.")


if __name__ == "__main__":
    main()

