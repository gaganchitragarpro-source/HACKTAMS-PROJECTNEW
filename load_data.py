import h5py
import numpy as np

HEALTHY_GROUPS = ['baseline_1', 'baseline_2']
DAMAGE_GROUPS  = ['damage_1', 'damage_2', 'damage_3', 'damage_4', 'damage_5', 'damage_6']
SAMPLE_RATE    = 128
WINDOW_SAMPLES = SAMPLE_RATE * 10
STEP_SAMPLES   = WINDOW_SAMPLES // 2

def sliding_windows(signal, window, step):
    starts = range(0, len(signal) - window + 1, step)
    return np.array([signal[s:s+window] for s in starts])

def load_group(f, group_name):
    all_windows = []
    group = f[group_name]['acceleration']
    for test_key in sorted(group.keys()):
        test = group[test_key]
        for sensor_key in sorted(test.keys()):
            sensor = test[sensor_key]
            for axis in ['y', 'z']:
                if axis in sensor:
                    sig = sensor[axis][0].astype(np.float32)
                    wins = sliding_windows(sig, WINDOW_SAMPLES, STEP_SAMPLES)
                    all_windows.append(wins)
    return np.vstack(all_windows)

print("Loading healthy data...")
with h5py.File('Rt345Bridge.h5', 'r') as f:
    healthy = np.vstack([load_group(f, g) for g in HEALTHY_GROUPS])

print("Loading damaged data...")
with h5py.File('Rt345Bridge.h5', 'r') as f:
    damaged = np.vstack([load_group(f, g) for g in DAMAGE_GROUPS])

np.save('healthy_data.npy', healthy)
np.save('damaged_data.npy', damaged)

print(f"\nPhase 2 Complete!")
print(f"Healthy shape: {healthy.shape}")
print(f"Damaged shape: {damaged.shape}")