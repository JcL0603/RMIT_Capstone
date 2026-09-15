# -*- coding: utf-8 -*-
"""
EEET2450 Capstone Project - SisFall Dataset Automated Multi-Class Preprocessing & Augmentation (v6.0)
Student: Jacky LIN (S4143087 / 247024328)
Supervisor: Dr. Scottie MAN (文子賢)

Description:
This script implements a "Data-Centric AI" strategy to resolve the severe misclassification
between 'class_static' and 'class_stairs'.
Key Enhancements in v6.0:
1. High-density sliding window oversampling for 'class_stairs' (stride reduced to 10 points, i.e., 80% overlap).
   This aggressively boosts stairs sample size by 500% from the root, giving the neural network balanced gradients.
2. Normal non-overlapping stride (50 points, 0% overlap) maintained for static and walk to prevent dataset explosion.
3. 25Hz downsampling, bits conversion to m/s², and foolproof 'label.filename.csv' structure maintained.
"""

import os
import glob
import numpy as np
import pandas as pd

# ==================== 1. Hardware & Physics Constants ====================
FS_RAW = 200            # Original SisFall sampling rate (200Hz)
FS_TARGET = 25          # Target TinyML sampling rate (25Hz)
DOWNSAMPLE_FACTOR = int(FS_RAW / FS_TARGET)  # Downsampling ratio (200/25 = 8)

RANGE = 16              # ADXL345 accelerometer range (+-16g)
RESOLUTION = 13         # ADXL345 13-bit resolution
ADC_TO_G = (2 * RANGE) / (2 ** RESOLUTION) 
G_TO_MS2 = 9.80665       # Standard gravity conversion (g to m/s^2)

# Unified Window size for training: 2.0 seconds = 50 samples at 25Hz
WINDOW_SAMPLES = 50     

# ==================== 2. Directories Setup ====================
RAW_DATA_DIR = r"E:\OneDrive - RMIT University\EEET2450 Research Methods for Engineers\S4143087 Shared with Scottie\Dataset"
OUTPUT_DIR = r"E:\OneDrive - RMIT University\EEET2450 Research Methods for Engineers\S4143087 Shared with Scottie\Dataset_25Hz_Sliced_v6"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("====================================================================")
print("  SisFall Automated 5-Class Preprocessing & Augmentation v6.0       ")
print("  Target Sampling Rate: 25Hz | Physical Units: m/s^2 & dps          ")
print("  💡 SPECIAL: 5x Oversampling Augmentation for Class_Stairs Enabled ")
print("  RMIT Capstone Project - Jacky LIN (Supervisor: Dr. Scottie Man)    ")
print("====================================================================")

# Recurse and search for all raw data txt files
file_paths = glob.glob(os.path.join(RAW_DATA_DIR, "**/*.txt"), recursive=True)
processed_count = 0
ignored_count = 0

if len(file_paths) == 0:
    print(f"⚠️ Warning: No raw files found in '{RAW_DATA_DIR}'.")
    print(f"Please check your path configuration.")
else:
    print(f"🔍 Found {len(file_paths)} raw files. Starting automated pipeline...")

# Activity mappings for ADL
walk_prefixes = ["D01", "D02", "D03", "D04", "D05"]
stairs_prefixes = ["D06", "D07"]
static_prefixes = ["D08", "D09", "D10", "D11", "D12", "D13"]

# Helper to inject monotonic timestamp (0, 40, 80, ..., 1960 ms)
def inject_timestamp_and_save(df, output_path):
    if len(df) != WINDOW_SAMPLES:
        if len(df) < WINDOW_SAMPLES:
            padding = pd.DataFrame([df.iloc[-1]] * (WINDOW_SAMPLES - len(df)))
            df = pd.concat([df, padding], ignore_index=True)
        else:
            df = df.iloc[:WINDOW_SAMPLES].reset_index(drop=True)
            
    df.insert(0, 'timestamp', [i * 40 for i in range(WINDOW_SAMPLES)])
    df.to_csv(output_path, index=False)

for path in file_paths:
    file_name = os.path.basename(path)
    if "Readme" in file_name or not file_name.endswith(".txt") or file_name.startswith("."):
        ignored_count += 1
        continue
        
    try:
        raw_df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3, 4, 5])
        
        ax_raw, ay_raw, az_raw = raw_df[0], raw_df[1], raw_df[2]
        gx_raw, gy_raw, gz_raw = raw_df[3], raw_df[4], raw_df[5]
        
        ax_converted = ax_raw * ADC_TO_G * G_TO_MS2
        ay_converted = ay_raw * ADC_TO_G * G_TO_MS2
        az_converted = az_raw * ADC_TO_G * G_TO_MS2
        gx_converted = gx_raw / 14.375
        gy_converted = gy_raw / 14.375
        gz_converted = gz_raw / 14.375
        
        clean_df = pd.DataFrame({
            'ax': ax_converted, 'ay': ay_converted, 'az': az_converted,
            'gx': gx_converted, 'gy': gy_converted, 'gz': gz_converted
        })
        
        ds_df = clean_df.iloc[::DOWNSAMPLE_FACTOR].reset_index(drop=True)
        prefix = file_name.split("_")[0]
        
        if file_name.startswith("F"):
            avm = np.sqrt(ds_df['ax']**2 + ds_df['ay']**2 + ds_df['az']**2)
            peak_idx = avm.idxmax()
            
            end_prefall = peak_idx - 3
            start_prefall = end_prefall - WINDOW_SAMPLES
            
            if start_prefall >= 0 and end_prefall < len(ds_df):
                prefall_df = ds_df.iloc[start_prefall:end_prefall].copy()
                out_name = f"class_prefall_alert.{file_name.replace('.txt', '_prefall.csv')}"
                inject_timestamp_and_save(prefall_df, os.path.join(OUTPUT_DIR, out_name))
                processed_count += 1
                
            start_fall = peak_idx
            end_fall = start_fall + WINDOW_SAMPLES
            
            if start_fall >= 0 and end_fall <= len(ds_df):
                fall_df = ds_df.iloc[start_fall:end_fall].copy()
                out_name = f"class_fall.{file_name.replace('.txt', '_fall.csv')}"
                inject_timestamp_and_save(fall_df, os.path.join(OUTPUT_DIR, out_name))
                processed_count += 1
                
        elif file_name.startswith("D"):
            label = ""
            stride = 50 # Default stride (0% overlap) for normal ADLs
            
            if prefix in walk_prefixes:
                label = "class_walk"
                stride = 50
            elif prefix in stairs_prefixes:
                label = "class_stairs"
                stride = 10 # 💡 CRITICAL AUGMENTATION: Stride of 10 points (80% overlap) generates 5x more data!
            elif prefix in static_prefixes:
                label = "class_static"
                stride = 50
            else:
                ignored_count += 1
                continue
                
            num_windows = (len(ds_df) - WINDOW_SAMPLES) // stride + 1
            if num_windows <= 0:
                num_windows = 1
            
            for w in range(num_windows):
                start_idx = w * stride
                end_idx = start_idx + WINDOW_SAMPLES
                
                # Bounds check
                if end_idx > len(ds_df):
                    end_idx = len(ds_df)
                    start_idx = len(ds_df) - WINDOW_SAMPLES
                    if start_idx < 0:
                        start_idx = 0
                
                sliced_df = ds_df.iloc[start_idx:end_idx].copy()
                
                out_name = f"{label}.{file_name.replace('.txt', f'_w{w}.csv')}"
                inject_timestamp_and_save(sliced_df, os.path.join(OUTPUT_DIR, out_name))
                processed_count += 1
                
    except Exception as e:
        print(f"❌ Error processing file '{file_name}': {str(e)}")

print("\n==================================================")
print(f"🎉 5-Class Multi-Class Slicing & Augmentation Completed!")
print(f"📂 Output Folder: {OUTPUT_DIR}")
print(f"✅ Sliced files created: {processed_count}")
print(f"🚫 Ignored/Other files: {ignored_count}")
print("==================================================")