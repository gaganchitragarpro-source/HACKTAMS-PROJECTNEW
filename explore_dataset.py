import h5py
import numpy as np

FILE_PATH = 'Rt345Bridge.h5'

print("=" * 60)
print("PHASE 1: DATASET EXPLORATION")
print("=" * 60)

with h5py.File(FILE_PATH, 'r') as f:

    # ── 1. Print every key and nested key in the file ──────────
    print("\n── Full File Structure ───────────────────────────────")
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}[DATASET] {name}")
            print(f"{indent}          shape={obj.shape} | "
                  f"dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}[GROUP]   {name}")
    f.visititems(print_structure)

    # ── 2. Print top-level attributes (often has sampling rate) ─
    print("\n── Top-Level Attributes ──────────────────────────────")
    if len(f.attrs) == 0:
        print("  No top-level attributes found.")
    for key, val in f.attrs.items():
        print(f"  {key}: {val}")

    # ── 3. Dig into every dataset and print attributes + sample ─
    print("\n── Per-Dataset Details ───────────────────────────────")
    def inspect_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"\n  >> {name}")
            print(f"     Shape       : {obj.shape}")
            print(f"     Dtype       : {obj.dtype}")
            # Print any attributes on this dataset (sampling rate lives here)
            if len(obj.attrs) > 0:
                print(f"     Attributes  :")
                for k, v in obj.attrs.items():
                    print(f"       {k}: {v}")
            else:
                print(f"     Attributes  : none")
            # Print first 5 values as a sanity check
            try:
                data = obj[()]
                if data.ndim == 1:
                    print(f"     First 5 vals: {data[:5]}")
                    print(f"     Min / Max   : {data.min():.4f} / "
                          f"{data.max():.4f}")
                    print(f"     Mean / Std  : {data.mean():.4f} / "
                          f"{data.std():.4f}")
                elif data.ndim == 2:
                    print(f"     First row   : {data[0, :5]}")
                    print(f"     Min / Max   : {data.min():.4f} / "
                          f"{data.max():.4f}")
            except Exception as e:
                print(f"     Could not read values: {e}")
    f.visititems(inspect_dataset)

    # ── 4. If there are multiple accelerometer channels, ────────
    #       find which one has the highest variance
    print("\n── Variance Check (find best accelerometer axis) ─────")
    accel_channels = {}
    def find_accels(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Flag anything that looks like an accelerometer
            lower = name.lower()
            if any(x in lower for x in ['accel', 'acc', 'ax', 'ay',
                                          'az', 'channel', 'ch']):
                try:
                    data = obj[()].flatten().astype(np.float64)
                    accel_channels[name] = np.var(data)
                except:
                    pass
    f.visititems(find_accels)

    if accel_channels:
        print("\n  Variance per accelerometer-like channel:")
        for name, var in sorted(accel_channels.items(),
                                key=lambda x: x[1], reverse=True):
            print(f"    {name:<50} variance = {var:.6f}")
        best = max(accel_channels, key=accel_channels.get)
        print(f"\n  ✓ Highest variance channel: {best}")
    else:
        print("  No channels with obvious accelerometer names found.")
        print("  Check the structure above and identify them manually.")

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("Read the output above and note:")
print("  1. Which keys are accelerometer channels")
print("  2. Which keys are pre-damage vs post-damage")
print("  3. What the sampling rate is")
print("  4. Which axis has highest variance")
print("=" * 60)