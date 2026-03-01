import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import os
import sys

# ── 1. Load the Pre-trained PyTorch Network and Scaler ────────

# We must define the EXACT same architecture that was saved in model.pth
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

import joblib

try:
    # Scale Data was saved using joblib, not raw pickle
    scaler = joblib.load('signal_scaler.pkl')
    print("✅ Loaded signal_scaler.pkl successfully.")
    
    # Initialize the model with 19 features (as per original pipeline)
    model = DenseAutoencoder(19)
    # The model was saved as a dict with 'state', 'dim', 'threshold', etc.
    checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state'])
    model.eval()
    
    # Override our hardcoded threshold with the actual one from the save file
    THRESHOLD = checkpoint.get('threshold', 0.8244)
    print(f"✅ Loaded Core PyTorch Autoencoder (model.pth) successfully. Threshold: {THRESHOLD:.4f}\n")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    print("Ensure 'signal_scaler.pkl' and 'model.pth' are in this directory.")
    sys.exit(1)

# (Pulled dynamically from checkpoint now)

# ── 2. The Feature Extraction Pipeline (Matches the edge device) ────────

import scipy.signal
# Physics Toolbox records at roughly 100-200Hz. Let's assume 128Hz for standard math.
SAMPLE_RATE = 128

def extract_features(window):
    """
    Exactly matches the 19 features our Pytorch Normalization expects.
    """
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-12)
    kurt = scipy.stats.kurtosis(window, nan_policy='omit')
    sk = scipy.stats.skew(window, nan_policy='omit')
    energy = np.sum(window**2)
    
    # Frequency features
    w_zero = window - np.mean(window)
    n = len(w_zero)
    hann = np.hanning(n)
    fft_vals = np.fft.rfft(w_zero * hann)
    freqs = np.fft.rfftfreq(n, d=1.0/SAMPLE_RATE)
    
    psd = (np.abs(fft_vals)**2) / (SAMPLE_RATE * n)
    dom_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
    freq_energy = np.sum(psd)
    
    bands = [(0.1, 5), (5, 20), (20, 40), (40, 50)]
    band_energies = []
    for fmin, fmax in bands:
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_energies.append(np.sum(psd[mask]) if np.any(mask) else 0)
        
    parseval = float(energy / (freq_energy + 1e-12))
    
    # Generate 4 dominant peaks
    peaks, _ = scipy.signal.find_peaks(psd, height=np.max(psd)*0.1)
    top_peaks = sorted(peaks, key=lambda x: psd[x], reverse=True)[:4]
    peak_freqs = [freqs[p] for p in top_peaks]
    peak_amps  = [psd[p] for p in top_peaks]
    
    # Pad if we didn't find 4 peaks
    while len(peak_freqs) < 4: peak_freqs.append(0)
    while len(peak_amps) < 4: peak_amps.append(0)
        
    feats = [
        rms, peak, crest, kurt, sk, energy, dom_freq, parseval,
        peak_freqs[0], peak_amps[0], peak_freqs[1], peak_amps[1],
        peak_freqs[2], peak_amps[2], peak_freqs[3], peak_amps[3],
        band_energies[0], band_energies[1], band_energies[2], band_energies[3]
    ]
    # Keep only the first 19 to match exactly
    return feats[:19]

def process_physics_toolbox_csv(csv_path):
    """
    Physics Toolbox creates a file with Time (s), gFx, gFy, gFz.
    We'll combine them to an absolute magnitude.
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None
        
    df = pd.read_csv(csv_path)
    
    # Try to find the acceleration columns. Physics toolbox uses gFx, gFy, gFz or ax, ay, az.
    col_x, col_y, col_z = None, None, None
    for c in df.columns:
        c_lower = c.lower()
        if ('acceleration' in c_lower and 'x' in c_lower) or 'fx' in c_lower or 'ax' in c_lower or 'x' == c_lower.strip(): col_x = c
        elif ('acceleration' in c_lower and 'y' in c_lower) or 'fy' in c_lower or 'ay' in c_lower or 'y' == c_lower.strip(): col_y = c
        elif ('acceleration' in c_lower and 'z' in c_lower) or 'fz' in c_lower or 'az' in c_lower or 'z' == c_lower.strip(): col_z = c
        
    if not col_x or not col_y or not col_z:
        print(f"❌ Could not identify X/Y/Z structural columns in {csv_path}")
        print(f"Columns found: {df.columns.tolist()}")
        return None
        
    mag = np.sqrt(df[col_x]**2 + df[col_y]**2 + df[col_z]**2)
    # Dynamically mean-center the signal to 0. 
    # This automatically handles BOTH sensors "With Gravity" (subtracts ~9.8 or ~1.0) and "Without Gravity" (subtracts ~0)
    mag = mag - np.mean(mag)
    
    # Slice the continuous signal into 10-second (1280 sample) chunks for the AI
    window_pts = 1280
    extracted_features = []
    
    for i in range(0, len(mag) - window_pts, int(window_pts/2)):
        window = mag[i : i+window_pts].values
        feats = extract_features(window)
        extracted_features.append(feats)
        
    if not extracted_features:
        print("❌ Not enough data! Need at least 10 seconds of continuous recording.")
        return None
        
    return np.array(extracted_features)

# ── 3. Run Inference ────────

def evaluate_recording(title, csv_path):
    print(f"--- 📡 Analyzing Hardware Signal: {title} ---")
    raw_feats = process_physics_toolbox_csv(csv_path)
    if raw_feats is None:
        return
        
    # Step 1: Normalize through the exact same scaler
    try:
        norm_feats = scaler.transform(raw_feats)
        X_tensor = torch.FloatTensor(norm_feats)
    except Exception as e:
        print(f"❌ Normalization Failed. The CSV might have missing data. Error: {e}")
        return
        
    # Step 2: Push through Autoencoder
    with torch.no_grad():
        reconstructed = model(X_tensor)
        # Calculate Mean Squared Error across all 19 features for every window
        mse = torch.mean((reconstructed - X_tensor)**2, dim=1).numpy()
        
    # Step 3: Grade the Bridge
    avg_mse = np.mean(mse)
    max_mse = np.max(mse)
    
    print(f"  > Scanned {len(mse)} vibration sequences.")
    print(f"  > Average PyTorch MSE: {avg_mse:.4f}  (Threshold = {THRESHOLD})")
    print(f"  > Maximum PyTorch MSE: {max_mse:.4f}")
    
    if avg_mse > THRESHOLD:
        print(f"\n  🚨 DAMAGE DETECTED! MSE exceeds structural threshold.")
        vibration_risk = min(100, int((avg_mse / THRESHOLD) * 50))
        print(f"  📊 Final Vibration Risk Score: {vibration_risk}/100")
    else:
        print(f"\n  ✅ SYSTEM HEALTHY. AI recognized structural resonance.")
        vibration_risk = int((avg_mse / THRESHOLD) * 50)
        print(f"  📊 Final Vibration Risk Score: {vibration_risk}/100")
        
    print("\n")

# If script is run directly, execute the dummy files
if __name__ == "__main__":
    if not os.path.exists("phone_healthy.csv"):
        print("Waiting for smartphone CSV data... (Please save 'phone_healthy.csv' and 'phone_damaged.csv')")
    else:
        evaluate_recording("Healthy Phone Baseline", "phone_healthy.csv")
        
    if os.path.exists("phone_damaged.csv"):
        evaluate_recording("Simulated Damage Test", "phone_damaged.csv")
