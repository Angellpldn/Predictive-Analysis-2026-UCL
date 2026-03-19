#!/usr/bin/env python3
"""
Download the Titanic dataset to the current directory.
This script fetches the dataset from a reliable source and saves it as titanic.csv
"""

import pandas as pd
import urllib.request
from io import StringIO
from pathlib import Path

def download_titanic():
    """Download Titanic dataset from GitHub"""
    
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    output_file = Path('titanic.csv')
    
    print(f"Downloading Titanic dataset from: {url}")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            csv_data = response.read().decode('utf-8')
        
        df = pd.read_csv(StringIO(csv_data))
        df.to_csv(output_file, index=False)
        print(f"✓ Successfully downloaded and saved to: {output_file}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"✗ Failed to download dataset: {e}")
        print("\nAlternative: Manually download from:")
        print("  https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        print("  Then save it as 'titanic.csv' in this directory")
        return False

if __name__ == '__main__':
    success = download_titanic()
    exit(0 if success else 1)
