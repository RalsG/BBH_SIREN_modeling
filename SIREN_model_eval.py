# standalone evaluation snippet — expects processed .npz containing params, time_axes, waveforms if available
import numpy as np
import torch
from SIREN import Siren
from SHAN import SHAN

# load processed arrays you saved with build_inputs_targets_from_loaded
d = np.load("processed_data\processed_1x1000_seed43.npz")  # or whatever file you have
inputs = d['inputs']          # flattened (n*T, p+1) where last column is time
targets = d['targets']        # flattened strain (n*T,)
params = d['params']          # (n, p)
times = d['time_axes']        # (n, T)
n, T = times.shape
p = params.shape[1]
in_features = p + 1

# build model and load checkpoint
# model = Siren(in_features=in_features, hidden_features=512, hidden_layers=5, out_features=1).eval()
model = SHAN(in_features=in_features, hidden_features=128, hidden_layers=10).eval()
ckpt = torch.load("checkpoint_models\SHAN_checkpoint_1waves_1000epochs_10x128.pt", map_location='cpu')
model.load_state_dict(ckpt['model_state'])

# reconstruct per-sim waveforms
recon = np.zeros((n, T), dtype=np.float32)
for i in range(n):
    # create input for this simulation: repeat params row and times
    param_row = np.repeat(params[i:i+1, :], T, axis=0)   # (T, p)
    inp = np.concatenate([param_row, times[i:i+1,:].T], axis=1).astype(np.float32)  # (T, p+1)
    with torch.no_grad():
        out = model(torch.from_numpy(inp)).numpy().reshape(-1)
    recon[i,:] = out

# compute metrics per simulation
true = targets.reshape(n, T)
mse_per = np.mean((recon - true)**2, axis=1)
rmse_per = np.sqrt(mse_per)
rms_true = np.sqrt(np.mean(true**2, axis=1))
nmse_per = mse_per / (rms_true**2 + 1e-12)   # normalized MSE
r2_per = 1.0 - mse_per / (np.var(true, axis=1) + 1e-12)

print("RMSE median, mean:", np.median(rmse_per), np.mean(rmse_per))
print("Normalized MSE median:", np.median(nmse_per))
print("R^2 median:", np.median(r2_per))

# save a few example plots for manual inspection (if matplotlib available)
import matplotlib.pyplot as plt
for i in [0, n//4, n//2, -1]:
#i = 0
    plt.figure(figsize=(6,3))
    plt.plot(times[i], true[i], label='true')
    plt.plot(times[i], recon[i], label='recon', alpha=0.8)
    plt.title(f"sim {i} RMSE={rmse_per[i]:.4e} NMSE={nmse_per[i]:.3e}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"SHAN_1wave_1000epochs_recon_example_{i}.png")