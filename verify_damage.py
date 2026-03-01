import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. ARCHITECTURE: Matching the layers in your model.pth (Fixes RuntimeError)
class BridgeAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# 2. DATA LOADING & MSE CALCULATION (Fixes NameError)
checkpoint = torch.load('model.pth')
h_scaled = np.load('healthy_scaled.npy')
d_scaled = np.load('damaged_scaled.npy')

model = BridgeAutoencoder(checkpoint['dim'])
model.load_state_dict(checkpoint['state'])
model.eval()

def get_mse(data):
    with torch.no_grad():
        x = torch.FloatTensor(data)
        recon = model(x)
        return torch.mean((x - recon)**2, dim=1).numpy()

h_mse = get_mse(h_scaled)
d_mse = get_mse(d_scaled)

# 3. RESEARCH-GRADE ANALYTICAL REPORT (Log-Scaled & Vectorized)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Log-Scaled Probability Density
# log=True makes the small healthy baseline visible despite the large anomaly count
ax1.hist(h_mse, bins=60, alpha=0.5, label='Healthy (Baseline)', color='#4A90E2', density=True, log=True)
ax1.hist(d_mse, bins=60, alpha=0.5, label='Damaged (Anomalies)', color='#E94E77', density=True, log=True)
ax1.axvline(checkpoint.get('threshold', 0.5), color='black', linestyle='--', linewidth=1.5, label='Alarm Threshold')

ax1.set_title('Anomaly Signature (Log-Normalized)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Reconstruction Error (MSE Score)', fontsize=12)
ax1.set_ylabel('Log-Probability Density', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, which="both", ls="-", alpha=0.1)

# Panel 2: Structural Feature Sensitivity (Diagnostic Report)
feature_names = ['RMS', 'Peak', 'Crest', 'Kurtosis', 'Skew', 'Std', 'Energy', 'Parseval', 
                 'F1_Freq', 'F2_Freq', 'F3_Freq', 'F1_Amp', 'F2_Amp', 'F3_Amp', 
                 'Band1', 'Band2', 'Band3', 'Band4', 'Dom_Freq']
importance = np.abs(np.mean(d_scaled, axis=0) - np.mean(h_scaled, axis=0))
sorted_idx = np.argsort(importance)

ax2.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color='#2E8B57')
ax2.set_title('Structural Feature Sensitivity', fontsize=14, fontweight='bold')
ax2.set_xlabel('Relative Magnitude of Change', fontsize=12)
ax2.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()

# 4. VECTOR SAVING: PDF allows for infinite zoom on your presentation board
plt.savefig('bridge_analytical_report.pdf', format='pdf')
print("\n✅ ANALYSIS COMPLETE: 'bridge_analytical_report.pdf' generated with vector graphics.")