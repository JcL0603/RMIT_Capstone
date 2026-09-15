# -*- coding: utf-8 -*-
"""
RMIT Capstone Project (Part A) - Custom Wearable CSV Dataset Processor & Plotter
Student: Jacky LIN (S4143087)
Supervisor: Dr. Scottie MAN (文子賢)

Description:
This script processes raw CSV files exported from the Web BLE Monitor (v5.4),
cleans the timestamps, computes signal vector magnitude (AVM), segmentizes 
the continuous 25Hz stream into 2.0s (50 samples) sliding windows, and plots
the multi-axis motion curves. This aligns your local physical testing 
with Edge Impulse training formats.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_and_plot_telemetry(csv_path, output_dir=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: Raw CSV file not found at {csv_path}")
        
    print(f"🔄 Processing custom dataset: {os.path.basename(csv_path)}")
    
    # 1. Load exported Web BLE CSV
    # Columns: Timestamp, AX, AY, AZ, GX, GY, GZ, AVM, Classifier, Confidence
    df = pd.read_csv(csv_path)
    
    # Remove any rows with NaN or incomplete records
    df = df.dropna().reset_index(drop=True)
    
    # 2. Extract and Convert Physical Axes
    # Convert Timestamp from text '[10:04:15 AM]' to continuous milliseconds (40ms steps)
    df['millisecond'] = [i * 40 for i in range(len(df))]
    
    # Compute Instantaneous Acc Vector Magnitude (AVM) as a safety sanity check
    # AVM = sqrt(ax^2 + ay^2 + az^2)
    computed_avm = np.sqrt(df['AX']**2 + df['AY']**2 + df['AZ']**2)
    df['Computed_AVM'] = computed_avm
    
    print(f"📊 Dataset stats: {len(df)} samples parsed (~{len(df)*40/1000:.2f} seconds of continuous tracking)")
    print(f"📈 Acceleration Peak: {df['AX'].abs().max():.2f} m/s^2, Computed AVM Max: {computed_avm.max():.2f} m/s^2")
    print(f"🔄 Gyroscope Rotation Peak: {df['GX'].abs().max():.2f} dps")
    
    # 3. Sliding Window Segmentation (2.0s Window = 50 Samples at 25Hz)
    WINDOW_SIZE = 50
    STRIDE = 10  # 80% overlap for aggressive custom training augmentation
    
    windows_created = 0
    if len(df) >= WINDOW_SIZE:
        num_windows = (len(df) - WINDOW_SIZE) // STRIDE + 1
        print(f"📦 Generating {num_windows} sliding windows (2.0s length, 80% overlap) for Edge Impulse ingestion...")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for w in range(num_windows):
                start_idx = w * STRIDE
                end_idx = start_idx + WINDOW_SIZE
                window_df = df.iloc[start_idx:end_idx].copy()
                
                # Format exactly to meet Edge Impulse CSV ingestion specs
                ei_format_df = pd.DataFrame({
                    'timestamp': [i * 40 for i in range(WINDOW_SIZE)],
                    'ax': window_df['AX'],
                    'ay': window_df['AY'],
                    'az': window_df['AZ'],
                    'gx': window_df['GX'],
                    'gy': window_df['GY'],
                    'gz': window_df['GZ']
                })
                
                # Auto-generate a descriptive training file name
                lbl = window_df['Classifier'].iloc[0].lower().replace(" ", "_")
                filename = f"custom_{lbl}_trial_{w}.csv"
                ei_format_df.to_csv(os.path.join(output_dir, filename), index=False)
                windows_created += 1
            print(f"✅ Successfully exported {windows_created} training-ready sliced CSVs to '{output_dir}'")
            
    # 4. Multi-Axis Waveform Visualization Plotter
    plt.use('Agg') # Headless safe plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    time_sec = df['millisecond'] / 1000.0
    
    # Subplot 1: Linear Acceleration (AX, AY, AZ)
    ax1.plot(time_sec, df['AX'], label='AX', color='#ef4444', alpha=0.85)
    ax1.plot(time_sec, df['AY'], label='AY', color='#10b981', alpha=0.85)
    ax1.plot(time_sec, df['AZ'], label='AZ', color='#3b82f6', alpha=0.85)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_title('Real-time 3-Axis Acceleration Stream')
    
    # Subplot 2: Rotational Angular Velocity (GX, GY, GZ)
    ax2.plot(time_sec, df['GX'], label='GX', color='#f59e0b', alpha=0.85)
    ax2.plot(time_sec, df['GY'], label='GY', color='#8b5cf6', alpha=0.85)
    ax2.plot(time_sec, df['GZ'], label='GZ', color='#06b6d4', alpha=0.85)
    ax2.set_ylabel('Angular Velocity (dps)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    ax2.set_title('Real-time 3-Axis Gyroscope Stream')
    
    # Subplot 3: Signal Vector Magnitude (AVM) vs AI Classification State
    ax3.plot(time_sec, df['Computed_AVM'], label='AVM Magnitude', color='#ec4899', linewidth=2)
    ax3.set_xlabel('Time (Seconds)')
    ax3.set_ylabel('Magnitude (m/s²)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left')
    
    # Secondary Y-Axis to plot classification indices over time
    ax4 = ax3.twinx()
    # Map class name to numeric index for visual tracking
    class_map = {'STATIC': 0, 'WALK': 1, 'STAIRS': 2, 'PRE-FALL': 3, 'FALL': 4, 'FALL CRASH': 4}
    df['Class_Idx'] = df['Classifier'].map(class_map).fillna(0)
    ax4.step(time_sec, df['Class_Idx'], where='post', color='#64748b', linestyle=':', label='AI State', alpha=0.7)
    ax4.set_ylabel('AI Classifier State Index', color='#64748b')
    ax4.tick_params(axis='y', labelcolor='#64748b')
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_yticklabels(['STATIC', 'WALK', 'STAIRS', 'PRE-FALL', 'FALL'])
    
    plt.tight_layout()
    
    if output_dir:
        plot_path = os.path.join(output_dir, 'custom_telemetry_waveforms.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🎨 Waveform plot successfully compiled & saved to '{plot_path}'")
        
    print("==========================================================")

# Simple Mock generator to allow running verification test
if __name__ == "__main__":
    # Create mock BLE CSV to verify script integrity
    mock_data = {
        'Timestamp': [f"[10:00:{i:02d} AM]" for i in range(100)],
        'AX': np.random.normal(0, 1.0, 100) + 9.8,
        'AY': np.random.normal(0, 0.5, 100),
        'AZ': np.random.normal(0, 0.5, 100),
        'GX': np.random.normal(0, 10.0, 100),
        'GY': np.random.normal(0, 10.0, 100),
        'GZ': np.random.normal(0, 5.0, 100),
        'AVM': [10.0] * 100,
        'Classifier': ['WALK' if i > 30 else 'STATIC' for i in range(100)],
        'Confidence': [0.95] * 100
    }
    mock_df = pd.DataFrame(mock_data)
    
    scratch_dir = "/workspace/scratch"
    os.makedirs(scratch_dir, exist_ok=True)
    mock_csv = os.path.join(scratch_dir, "mock_telemetry_dataset.csv")
    mock_df.to_csv(mock_csv, index=False)
    
    # Dry run test
    process_and_plot_telemetry(mock_csv, output_dir=scratch_dir)
