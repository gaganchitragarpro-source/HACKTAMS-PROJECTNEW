import os
import json
import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
import joblib
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── 1. PyTorch Model Initialization ──────────────────────────────────────────────
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

try:
    scaler = joblib.load('signal_scaler.pkl')
    model = DenseAutoencoder(19)
    checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state'])
    model.eval()
    THRESHOLD = checkpoint.get('threshold', 0.8244)
    print("✅ Loaded Core AI Model successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    scaler, model = None, None
    THRESHOLD = 0.8244

# ── 2. Signal Processing ───────────────────────────────────────────────────────
SAMPLE_RATE = 128

def extract_features(window):
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-12)
    kurt = scipy.stats.kurtosis(window, nan_policy='omit')
    sk = scipy.stats.skew(window, nan_policy='omit')
    energy = np.sum(window**2)
    
    w_zero = window - np.mean(window)
    n = len(w_zero)
    fft_vals = np.fft.rfft(w_zero * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0/SAMPLE_RATE)
    psd = (np.abs(fft_vals)**2) / (SAMPLE_RATE * n)
    dom_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
    freq_energy = np.sum(psd)
    
    bands = [(0.1, 5), (5, 20), (20, 40), (40, 50)]
    band_energies = [np.sum(psd[(freqs >= fmin) & (freqs <= fmax)]) for fmin, fmax in bands]
    parseval = float(energy / (freq_energy + 1e-12))
    
    peaks, _ = scipy.signal.find_peaks(psd, height=np.max(psd)*0.1)
    top_peaks = sorted(peaks, key=lambda x: psd[x], reverse=True)[:4]
    peak_freqs = [freqs[p] for p in top_peaks] + [0,0,0,0]
    peak_amps  = [psd[p] for p in top_peaks] + [0,0,0,0]
        
    feats = [rms, peak, crest, kurt, sk, energy, dom_freq, parseval] + \
            peak_freqs[:4] + peak_amps[:4] + band_energies
    return feats[:19]

def process_csv(filepath):
    df = pd.read_csv(filepath)
    col_x, col_y, col_z = None, None, None
    for c in df.columns:
        c_lower = c.lower()
        if 'fx' in c_lower or 'ax' in c_lower or 'x' == c_lower.strip(): col_x = c
        elif 'fy' in c_lower or 'ay' in c_lower or 'y' == c_lower.strip(): col_y = c
        elif 'fz' in c_lower or 'az' in c_lower or 'z' == c_lower.strip(): col_z = c
        
    if not col_x or not col_y or not col_z: return None, None, None
        
    mag = np.sqrt(df[col_x]**2 + df[col_y]**2 + df[col_z]**2) - 1.0
    wins = [mag[i:i+1280].values for i in range(0, len(mag)-1280, 640)]
    if not wins: return None, None, None
    
    features = [extract_features(w) for w in wins]
    
    # Representative FFT
    med_idx = len(wins)//2
    w_med = wins[med_idx] - np.mean(wins[med_idx])
    n = len(w_med)
    fft_vals = np.fft.rfft(w_med * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0/SAMPLE_RATE)
    psd = (np.abs(fft_vals)**2) / (SAMPLE_RATE * n)
    
    return np.array(features), freqs.tolist(), psd.tolist()

# ── 3. Web Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'healthy_file' not in request.files or 'damaged_file' not in request.files:
        return redirect(url_for('index'))
        
    healthy_file = request.files['healthy_file']
    damaged_file = request.files['damaged_file']
    visual_score = float(request.form.get('visual_score', 0))
    
    h_path = os.path.join(app.config['UPLOAD_FOLDER'], 'healthy.csv')
    d_path = os.path.join(app.config['UPLOAD_FOLDER'], 'damaged.csv')
    healthy_file.save(h_path)
    damaged_file.save(d_path)
    
    # Process
    h_feats, h_f, h_p = process_csv(h_path)
    d_feats, d_f, d_p = process_csv(d_path)
    
    if h_feats is None or d_feats is None:
        return "Error parsing CSV columns. Must contain X, Y, Z axes.", 400
        
    # Infer
    with torch.no_grad():
        h_norm = scaler.transform(h_feats)
        d_norm = scaler.transform(d_feats)
        
        h_tensor = torch.FloatTensor(h_norm)
        d_tensor = torch.FloatTensor(d_norm)
        
        h_mse = torch.mean((model(h_tensor) - h_tensor)**2, dim=1).numpy()
        d_mse = torch.mean((model(d_tensor) - d_tensor)**2, dim=1).numpy()
        
    avg_h_mse = float(np.mean(h_mse))
    avg_d_mse = float(np.mean(d_mse))
    
    # Calculate Risk
    vib_risk = min(100, (avg_d_mse / THRESHOLD) * 50)
    final_risk = (vib_risk * 0.65) + (visual_score * 0.35)
    
    # Pack for UI
    report_data = {
        'threshold': THRESHOLD,
        'avg_h_mse': avg_h_mse,
        'avg_d_mse': avg_d_mse,
        'vib_risk': round(vib_risk, 1),
        'vis_risk': visual_score,
        'final_risk': round(final_risk, 1),
        'h_mse_dist': h_mse.tolist(),
        'd_mse_dist': d_mse.tolist(),
        'h_f': h_f[:300], 'h_p': h_p[:300],
        'd_f': d_f[:300], 'd_p': d_p[:300],
        'h_parseval': h_feats[:, 7].tolist(),
        'd_parseval': d_feats[:, 7].tolist(),
    }
    
    return render_template('report.html', data=json.dumps(report_data), d=report_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
