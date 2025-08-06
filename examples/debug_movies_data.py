#!/usr/bin/env python3
"""
Debug script to understand the MoviesData.csv structure from PyMC.
"""

import pandas as pd
import pymc as pm

print("🔍 DEBUGGING MOVIES DATA STRUCTURE")
print("=" * 50)

# Get the raw data
movies_raw = pm.get_data("MoviesData.csv")
print(f"Raw data type: {type(movies_raw)}")
print(f"Raw data length: {len(movies_raw)}")
print(f"Raw data preview (first 500 chars):")
print(movies_raw[:500])

# Try different parsing approaches
print("\n" + "="*50)
print("APPROACH 1: Direct DataFrame")
try:
    movies1 = pd.DataFrame(movies_raw)
    print("✓ Direct DataFrame approach:")
    print(f"Shape: {movies1.shape}")
    print(f"Columns: {list(movies1.columns)}")
    print("First few rows:")
    print(movies1.head())
except Exception as e:
    print(f"✗ Direct DataFrame failed: {e}")

print("\n" + "="*50)
print("APPROACH 2: StringIO parsing")
try:
    import io
    movies2 = pd.read_csv(io.StringIO(movies_raw.decode('utf-8')))
    print("✓ StringIO parsing approach:")
    print(f"Shape: {movies2.shape}")
    print(f"Columns: {list(movies2.columns)}")
    print("First few rows:")
    print(movies2.head())
except Exception as e:
    print(f"✗ StringIO parsing failed: {e}")

print("\n" + "="*50)
print("APPROACH 3: BytesIO parsing")
try:
    import io
    movies3 = pd.read_csv(io.BytesIO(movies_raw))
    print("✓ BytesIO parsing approach:")
    print(f"Shape: {movies3.shape}")
    print(f"Columns: {list(movies3.columns)}")
    print("First few rows:")
    print(movies3.head())
except Exception as e:
    print(f"✗ BytesIO parsing failed: {e}")

print("\n" + "="*50)
print("APPROACH 4: Manual parsing")
try:
    # Split by lines and parse manually
    lines = movies_raw.decode('utf-8').split('\n')
    print("First few lines:")
    for i, line in enumerate(lines[:5]):
        print(f"Line {i}: {line}")
    
    # Parse header
    header = lines[0].strip().split(',')
    print(f"\nHeader: {header}")
    
    # Parse data rows
    data_rows = []
    for line in lines[1:]:
        if line.strip():
            data_rows.append(line.strip().split(','))
    
    print(f"Number of data rows: {len(data_rows)}")
    print("First data row:", data_rows[0] if data_rows else "No data")
    
except Exception as e:
    print(f"✗ Manual parsing failed: {e}") 