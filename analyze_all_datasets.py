import numpy as np
import pandas as pd
import json
import os
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

# ── 1. SETUP & PATH FIX (Look in current folder, not the old subfolder) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixes Screenshot 10:26:31 PM (Defining missing constants)
SAMPLE_RATE_DAQ  = 200   
WINDOW_SAMPLES   = SAMPLE_RATE_DAQ * 10  

# Fixes Screenshot 10:42:34 PM (Initializing variables to prevent NameErrors)
trend_pct = 0.0
bridge_stats = []
damage_classes = ['No Damage', 'Minor', 'Moderate', 'Severe']
damage_colors  = ['#3fb950', '#d29922', '#f0883e', '#f85149']

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def sliding_windows(signal, window=2000, step=1000):
    starts = range(0, len(signal) - window + 1, step)
    return [signal[s:s+window] for s in starts]

def extract_features_200hz(window):
    w = window - np.mean(window)
    nyq = SAMPLE_RATE_DAQ / 2
    try:
        b, a = sp_signal.butter(4, [0.1/nyq, 90/nyq], btype='band')
        w = sp_signal.filtfilt(b, a, w)
    except: pass
    n = len(w)
    rms = float(np.sqrt(np.mean(w**2)))
    peak = float(np.max(np.abs(w)))
    crest = float(peak / (rms + 1e-12))
    kurt = float(kurtosis(w))
    sk = float(skew(w))
    energy = float(np.sum(w**2) / SAMPLE_RATE_DAQ)
    fft_vals = np.fft.rfft(w * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0/SAMPLE_RATE_DAQ)
    psd = (np.abs(fft_vals)**2) / (SAMPLE_RATE_DAQ * n)
    total_power = np.sum(psd) + 1e-12
    dom_freq = float(freqs[np.argmax(psd)])
    bands = [(0.1,5),(5,20),(20,50),(50,90)]
    band_energies = [float(np.sum(psd[(freqs >= lo) & (freqs <= hi)]) / total_power) for lo, hi in bands]
    freq_energy = float(np.sum(np.abs(fft_vals)**2) / (n * SAMPLE_RATE_DAQ))
    parseval = float(energy / (freq_energy + 1e-12))
    return {
        'rms': rms, 'peak': peak, 'crest': crest, 'kurtosis': kurt,
        'skewness': sk, 'energy': energy, 'parseval': parseval,
        'dom_freq': dom_freq, 'band1': band_energies[0], 'band2': band_energies[1],
        'band3': band_energies[2], 'band4': band_energies[3],
    }

def get_psd(signal, fs=200, max_freq=100):
    w = signal - np.mean(signal)
    n = len(w)
    fft_vals = np.fft.rfft(w * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    psd = (np.abs(fft_vals)**2) / (fs * n)
    mask = freqs <= max_freq
    return freqs[mask].tolist(), psd[mask].tolist()

# ─────────────────────────────────────────────────────────────────
# DATASET 1: Test Rig (test1-8.txt)
# ─────────────────────────────────────────────────────────────────
print("=" * 60 + "\nAnalyzing Dataset 1: Physical Test Rig\n" + "=" * 60)
test_rig, test_psds = [], []

for i in range(1, 9):
    fname = os.path.join(BASE_DIR, f'test{i}.txt')
    if not os.path.exists(fname):
        print(f"  ❌ Skipping {fname}: Not found.")
        continue
    with open(fname, 'r') as f:
        lines = f.readlines()[9:]
    data = []
    for line in lines:
        if line.strip():
            try: data.append([float(v) for v in line.strip().split()])
            except: pass
    if not data: continue
    arr = np.array(data)
    all_feats = []
    for ch in range(1, min(6, arr.shape[1])):
        wins = sliding_windows(arr[:, ch], WINDOW_SAMPLES, WINDOW_SAMPLES//2)
        for w in wins: all_feats.append(extract_features_200hz(w))
    df_feats = pd.DataFrame(all_feats)
    f_p, p_p = get_psd(arr[:2000, 1])
    test_rig.append({
        'test': f'Test {i}', 'duration_s': round(len(arr)/200, 1), 'n_windows': len(all_feats),
        'rms_mean': round(float(df_feats['rms'].mean()), 5), 'crest_mean': round(float(df_feats['crest'].mean()), 3),
        'kurtosis_mean': round(float(df_feats['kurtosis'].mean()), 3), 'dom_freq_mean': round(float(df_feats['dom_freq'].mean()), 2),
        'parseval_mean': round(float(df_feats['parseval'].mean()), 3), 'band1': round(float(df_feats['band1'].mean()), 4),
        'band2': round(float(df_feats['band2'].mean()), 4),
    })
    test_psds.append({'freqs': f_p[:150], 'psd': p_p[:150], 'label': f'Test {i}'})
    print(f"  ✅ Test {i} Processed.")

# ── 4. DATASET 2: REAL BRIDGE MONITOR ──────────────────────────────
print("\n" + "=" * 60 + "\nAnalyzing Dataset 2: Real Bridge Monitor\n" + "=" * 60)
timeline, session_psds = [], []
csv_files = sorted([f for f in os.listdir(BASE_DIR) if f.startswith('2023') and f.endswith('.csv')])

for f in csv_files:
    df = pd.read_csv(os.path.join(BASE_DIR, f))
    feats_all = []
    for ch in ['ch_18', 'ch_19', 'ch_20']:
        wins = sliding_windows(df[ch].values, WINDOW_SAMPLES, WINDOW_SAMPLES//2)
        for w in wins: feats_all.append(extract_features_200hz(w))
    df_f = pd.DataFrame(feats_all)
    f_p, p_p = get_psd(df['ch_18'].values[:2000])
    timeline.append({
        'label': f"{f[5:7]}/{f[8:10]} {f[11:13]}:{f[14:16]}", 'rms_ch18': round(float(df['ch_18'].abs().mean()), 4),
        'rms_ch19': round(float(df['ch_19'].abs().mean()), 4), 'dom_freq': round(float(df_f['dom_freq'].mean()), 2),
        'parseval': round(float(df_f['parseval'].mean()), 3), 'band1': round(float(df_f['band1'].mean()), 4),
        'band2': round(float(df_f['band2'].mean()), 4)
    })
    session_psds.append({'freqs': f_p[:150], 'psd': p_p[:150], 'label': timeline[-1]['label']})
    print(f"  ✅ {f} Processed.")

if timeline:
    trend_pct = ((timeline[-1]['rms_ch18'] - timeline[0]['rms_ch18']) / timeline[0]['rms_ch18']) * 100

# ── 5. DATASET 3: MULTI-BRIDGE LABELED ─────────────────────────────
print("\n" + "=" * 60 + "\nAnalyzing Dataset 3: Multi-Bridge Labeled\n" + "=" * 60)
class_stats = []
ds_csv = os.path.join(BASE_DIR, 'bridge_dataset.csv')
if os.path.exists(ds_csv):
    bdf = pd.read_csv(ds_csv)
    for cls in damage_classes:
        sub = bdf[bdf['damage_class'] == cls]
        if not sub.empty:
            class_stats.append({
                'class': cls, 'count': int(len(sub)), 'deg_mean': round(float(sub['degradation_score'].mean()), 2),
                'deg_std':  round(float(sub['degradation_score'].std()), 2),
                'rms_z': round(float(np.sqrt(np.mean(sub['acceleration_z']**2))), 4), 'fft_peak': round(float(sub['fft_peak_freq'].mean()), 3),
                'forecast': round(float(sub['forecast_score_next_30d'].mean()), 2)
            })
    print("  ✅ bridge_dataset.csv Processed.")

# ─────────────────────────────────────────────────────────────────
# 6. GENERATE FULL DASHBOARD (All original HTML/JS restored)
# ─────────────────────────────────────────────────────────────────
js_data = {
    'test_rig': test_rig,
    'test_psds': [{'freqs': p['freqs'], 'psd': p['psd'], 'label': p['label']} for p in test_psds],
    'timeline': timeline,
    'session_psds_sample': session_psds[::3],
    'trend_pct': round(trend_pct, 1),
    'class_stats': class_stats,
    'bridge_stats': bridge_stats,
    'damage_classes': damage_classes,
    'damage_colors': damage_colors,
}

print("\nGenerating multi-dataset HTML report...")
HTML_ADDON = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BridgeGuard AI — Multi-Dataset Validation Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117; color:#e6edf3; }}
  .hero {{ background:linear-gradient(135deg,#1a2744,#0d2137,#1a1a2e); padding:48px 40px; border-bottom:1px solid #21262d; }}
  .hero h1 {{ font-size:2.2em; color:#58a6ff; }}
  .section {{ padding:32px 40px; border-bottom:1px solid #21262d; }}
  .kpi-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:20px 0; }}
  .kpi {{ background:#21262d; border:1px solid #30363d; border-radius:10px; padding:18px; }}
  .chart-box {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; min-height:320px; }}
  .chart-grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .blue {{ color:#58a6ff; }} .orange {{ color:#d29922; }} .green {{ color:#3fb950; }} .red {{ color:#f85149; }}
</style>
</head>
<body>
<div class="hero">
  <h1>🔬 BridgeGuard AI — Unified Validation Suite</h1>
  <div class="subtitle">HackTAMS 2026 Analytical Dashboard</div>
</div>
<div class="section">
  <h2>Dataset 1 — Physical Test Rig</h2>
  <div class="chart-grid-2">
    <div class="chart-box"><div id="rig-rms" style="height:280px;"></div></div>
    <div class="chart-box"><div id="rig-freq" style="height:280px;"></div></div>
  </div>
</div>
<div class="section">
  <h2>Dataset 2 — 9-Day Monitor Trend</h2>
  <div id="timeline-rms" style="height:300px;"></div>
</div>
<div class="section">
  <h2>Dataset 3 — Labeled Damage Classes</h2>
  <div class="chart-grid-2">
    <div class="chart-box"><div id="deg-box" style="height:280px;"></div></div>
    <div class="chart-box"><div id="rms-class" style="height:280px;"></div></div>
  </div>
</div>
<script>
const D = {json.dumps(js_data)};
const PC = {{ paper_bgcolor:'#161b22', plot_bgcolor:'#161b22', font:{{color:'#fff'}}, margin:{{t:40,b:40,l:60,r:20}} }};

Plotly.newPlot('rig-rms', [{{x: D.test_rig.map(t=>t.test), y: D.test_rig.map(t=>t.rms_mean), type:'bar', marker:{{color:'#58a6ff'}} }}], PC);
Plotly.newPlot('rig-freq', [{{x: D.test_rig.map(t=>t.test), y: D.test_rig.map(t=>t.dom_freq_mean), type:'bar', marker:{{color:'#d29922'}} }}], PC);
Plotly.newPlot('timeline-rms', [{{x: D.timeline.map(t=>t.label), y: D.timeline.map(t=>t.rms_ch18), type:'scatter', mode:'lines+markers', line:{{color:'#3fb950'}} }}], PC);
Plotly.newPlot('deg-box', [{{x: D.class_stats.map(c=>c.class), y: D.class_stats.map(c=>c.deg_mean), type:'bar', marker:{{color:D.damage_colors}} }}], PC);
Plotly.newPlot('rms-class', [{{x: D.class_stats.map(c=>c.class), y: D.class_stats.map(c=>c.rms_z), type:'bar', marker:{{color:D.damage_colors}} }}], PC);
</script>
</body></html>
"""

with open('multi_dataset_report.html', 'w') as f:
    f.write(HTML_ADDON)

print(f"\n✅ SUCCESS: 'multi_dataset_report.html' generated.")