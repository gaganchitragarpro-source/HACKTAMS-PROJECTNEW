import torch
import torch.nn as nn
import numpy as np
import os

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

try:
    X_train = np.load('healthy_scaled.npy')
    # Replace NaNs just in case
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_tensor = torch.FloatTensor(X_train)
    print(f"Loaded {X_train.shape[0]} windows with {X_train.shape[1]} features.")

    model = BridgeAutoencoder(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Training Autoencoder for 100 epochs...")
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), X_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f}")

    # Compute real 95th percentile threshold on healthy data
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        h_mse = torch.mean((X_tensor - recon)**2, dim=1).numpy()

    threshold = float(np.percentile(h_mse, 95))
    print(f"\n95th percentile threshold: {threshold:.6f}")
    print(f"Healthy MSE mean: {h_mse.mean():.6f}, std: {h_mse.std():.6f}")

    torch.save({
        'state': model.state_dict(),
        'dim': X_train.shape[1],
        'threshold': threshold,
        'h_mse_mean': float(h_mse.mean()),
        'h_mse_std': float(h_mse.std()),
    }, "model.pth")

    print(f"\n✅ model.pth saved with threshold={threshold:.6f}")
    print(f"   Size: {os.path.getsize('model.pth')} bytes")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ CRASHED: {e}")