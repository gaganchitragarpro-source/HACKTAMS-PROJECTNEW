import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

SAMPLE_RATE = 128
NYQUIST = SAMPLE_RATE / 2

def bandpass(data, low, high, fs=SAMPLE_RATE, order=4):
    b, a = sp_signal.butter(order, [low/NYQUIST, high/NYQUIST], btype='band')
    return sp_signal.filtfilt(b, a, data)

def extract_features(window):
    # ── Phase 3: Cleaning ──────────────────────────────────────────
    w_zero_mean = window - np.mean(window)
    w = bandpass(w_zero_mean, 0.1, 50) 
    n = len(w)
    features = []

    # ── Phase 5: Time Domain (7 features) ──────────────────────────
    rms = np.sqrt(np.mean(w**2))
    time_energy = np.sum(w**2) / SAMPLE_RATE
    features.extend([rms, np.max(np.abs(w)), float(np.max(np.abs(w))/(rms+1e-12)), 
                     float(kurtosis(w)), float(skew(w)), np.std(w), time_energy])

    # ── Phase 5: FFT & PSD ─────────────────────────────────────────
    hanning_win = np.hanning(n)
    fft_vals = np.fft.rfft(w * hanning_win)
    freqs = np.fft.rfftfreq(n, d=1.0/SAMPLE_RATE)
    psd = (np.abs(fft_vals)**2) / (SAMPLE_RATE * n)

    # ── Parseval's Ratio (Feature 8) ──────────────────────────────
    freq_energy = np.sum(np.abs(fft_vals)**2) / (n * SAMPLE_RATE)
    features.append(float(time_energy / (freq_energy + 1e-12)))

    # ── Modal Peaks (Features 9-14) ────────────────────────────────
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(psd, height=np.mean(psd) * 2.0)
    top3_f, top3_a = [0.0]*3, [0.0]*3
    if len(peaks) > 0:
        idx = np.argsort(psd[peaks])[-3:]
        for i, p_idx in enumerate(idx):
            top3_f[i], top3_a[i] = freqs[peaks[p_idx]], psd[peaks[p_idx]]
    features.extend(top3_f + top3_a)

    # ── Band Ratios (Features 15-18) ──────────────────────────────
    total_power = np.sum(psd) + 1e-12
    for low, high in [(0.1, 5), (5, 20), (20, 40), (40, 50)]:
        mask = (freqs >= low) & (freqs <= high)
        features.append(float(np.sum(psd[mask]) / total_power))

    # ── Dominant Freq (Feature 19) ────────────────────────────────
    features.append(float(freqs[np.argmax(psd)]))

    return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    healthy_raw = np.load('healthy_data.npy')
    damaged_raw = np.load('damaged_data.npy')
    print(f"Extracting 19+ features from {len(healthy_raw)} windows...")
    h_feat = np.array([extract_features(w) for w in healthy_raw])
    d_feat = np.array([extract_features(w) for w in damaged_raw])
    np.save('healthy_features.npy', h_feat)
    np.save('damaged_features.npy', d_feat)
    print(f"✅ SUCCESS: New features per window: {h_feat.shape[1]}")