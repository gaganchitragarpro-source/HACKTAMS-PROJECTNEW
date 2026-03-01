import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

h_feat = np.load('healthy_features.npy')
d_feat = np.load('damaged_features.npy')

scaler = StandardScaler()
h_scaled = scaler.fit_transform(h_feat)
d_scaled = scaler.transform(d_feat)

np.save('healthy_scaled.npy', h_scaled)
np.save('damaged_scaled.npy', d_scaled)
joblib.dump(scaler, 'signal_scaler.pkl')
print(f"Scaled {h_scaled.shape[0]} windows and saved signal_scaler.pkl")