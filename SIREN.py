import sxs
import numpy as np
import pandas as pd
import scipy
import sys
import inspect
import time
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.amp import autocast, GradScaler
from SIREN import Siren # custom SIREN implementation
import argparse
from pathlib import Path

def get_sxs_simulation_list(num_simulations, eccentricity: str = 'noneccentric') -> pd.DataFrame:
    df = sxs.load("dataframe", tag="3.0.0")
    df_good = getattr(df.undeprecated.BBH, eccentricity)
    if df_good.shape[0] < num_simulations:
        print(f"Warning: Requested {num_simulations} simulations but only {df_good.shape[0]} available.")
    elif df_good.shape[0] > num_simulations:
        print(f"Selecting {num_simulations} simulations randomly from {df_good.shape[0]} available.")
        df_good = df_good.sample(n=num_simulations, random_state=42)
    else:
        print(f"Using all {num_simulations} available simulations.")
    
    start_time = time.time()
    # select scalar columns and expand spin vectors into components
    base_cols = ['reference_mass_ratio', 'initial_separation'] # add 'reference_eccentricity' if using eccentric simulations
    df_params = df_good[base_cols].copy()

    # expand spin vectors into x,y,z columns
    spin1 = pd.DataFrame(df_good['initial_dimensionless_spin1'].tolist(),
                         index=df_good.index,
                         columns=['initial_spin1_x', 'initial_spin1_y', 'initial_spin1_z'])
    spin2 = pd.DataFrame(df_good['initial_dimensionless_spin2'].tolist(),
                         index=df_good.index,
                         columns=['initial_spin2_x', 'initial_spin2_y', 'initial_spin2_z'])

    df_params = pd.concat([df_params, spin1, spin2], axis=1)
    end_time = time.time()
    print(f"Parameter DataFrame constructed in {end_time - start_time:.2f}")

    start_time_vec = np.asarray(df_good['reference_time'].tolist())

    return df_params, start_time_vec

def load_sxs_data(sxs_id_list: int, start_time_vec: int, time_axis_size: int = 1000, dom_em: int  = 2, dom_ell: int = 2, cache_dir: str = 'sxs_cache', force_reload: bool = False):
    """Load and interpolate SXS dominant mode strain for a list of simulation ids.

    This function caches per-simulation interpolated waveforms as compressed .npz files in `cache_dir`.
    If a cache file exists and force_reload is False it will be loaded instead of re-downloading/interpolating.

    Returns:
      waveforms: np.ndarray shape (n_sims, time_axis_size) -- real strain values
      time_axes: np.ndarray shape (n_sims, time_axis_size) -- time axes used for each sim
    """
    n = len(sxs_id_list)
    waveforms = np.zeros((n, time_axis_size), dtype=np.float32)
    time_axes = np.zeros((n, time_axis_size), dtype=np.float32)

    os.makedirs(cache_dir, exist_ok=True)

    for idx, sxs_id_str in enumerate(sxs_id_list):
        print(f"Loading simulation: {sxs_id_str} ({idx+1}/{n})")
        # safe filename
        safe_name = str(sxs_id_str).replace('/', '_').replace(':', '_')
        cache_path = os.path.join(cache_dir, f"{safe_name}.npz")

        if os.path.exists(cache_path) and not force_reload:
            try:
                d = np.load(cache_path)
                waveforms[idx, :] = d['waveform']
                time_axes[idx, :] = d['time']
                continue
            except Exception:
                print(f"Warning: failed to load cache {cache_path}, will re-create it.")

        try:
            simulation = sxs.load(sxs_id_str, download=True, progress=False)
        except Exception as e:
            print(f"Error loading simulation {sxs_id_str}: {e}")
            raise

        strain_modes_obj = getattr(simulation, 'h', None)
        if strain_modes_obj is None:
            raise ValueError(f"Strain not found or empty for simulation {sxs_id_str}.")

        peak_strain_time = strain_modes_obj.max_norm_time()
        time_axis = np.linspace(start_time_vec[idx], peak_strain_time, time_axis_size)

        # Extract dominant (ell, m) mode and take real part
        dom_mode = strain_modes_obj[:, strain_modes_obj.index(dom_ell, dom_em)]
        interpolated = dom_mode.interpolate(time_axis)
        wf = np.asarray(interpolated.ndarray.real, dtype=np.float32)
        waveforms[idx, :] = wf
        time_axes[idx, :] = time_axis.astype(np.float32)

        # save to cache
        try:
            np.savez_compressed(cache_path, waveform=wf, time=time_axis)
        except Exception as e:
            print(f"Warning: failed to write cache {cache_path}: {e}")

    return waveforms, time_axes


def train_siren_arrays(inputs: np.ndarray,
                       targets: np.ndarray,
                       in_features: int,
                       hidden_features: int = 256,
                       hidden_layers: int = 3,
                       out_features: int = 1,
                       epochs: int = 20,
                       batch_size: int = 4096,
                       lr: float = 1e-4,
                       device: str = None,
                       checkpoint_path: str = 'siren_checkpoint.pt') -> Siren:
    """Train a SIREN model from numpy inputs/targets.

    inputs: (N, D) array where D == in_features (e.g. [params..., time])
    targets: (N,) or (N,1) array of scalar strain values
    Returns trained model (also saves checkpoint each epoch).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"train_siren_arrays: device={device}, inputs={inputs.shape}, targets={targets.shape}")

    # Ensure shapes
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    tensor_x = torch.from_numpy(inputs.astype(np.float32))
    tensor_y = torch.from_numpy(targets.astype(np.float32))
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = Siren(in_features=in_features, hidden_features=hidden_features, hidden_layers=hidden_layers, out_features=out_features).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            with autocast(enabled=(device != 'cpu')):
                preds = model(xb)
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            batches += 1

        avg = running / max(1, batches)
        print(f"Epoch {epoch}/{epochs}  avg_loss={avg:.6e}")
        # checkpoint
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': opt.state_dict()}, checkpoint_path)

    print(f"Finished training, final checkpoint saved to {checkpoint_path}")
    return model


def build_inputs_targets_from_loaded(simulations_df: pd.DataFrame,
                                     start_time_vec: np.ndarray,
                                     loaded_strain,
                                     time_axes: np.ndarray = None,
                                     save_npz: str = None):
    """Build flattened inputs/targets suitable for train_siren_arrays.

    Parameters
    - simulations_df: DataFrame of parameters (rows correspond to simulations)
    - start_time_vec: array-like of start/reference times per simulation
    - loaded_strain: one of:
        * tuple (waveforms, time_axes) where waveforms shape (n, T)
        * ndarray shape (n, T, 2) where [:, :, 0] = strain, [:, :, 1] = time
        * ndarray shape (n, T) of strain (requires time_axes or will be approximated)
    - time_axes: optional ndarray (n, T) if needed
    - save_npz: optional path to save compressed arrays (inputs, targets, params, time_axes)

    Returns (inputs, targets, params_np, time_axes)
    """
    params_np = simulations_df.to_numpy(dtype=np.float32)
    n_sims = params_np.shape[0]

    # normalize input forms
    if isinstance(loaded_strain, (tuple, list)):
        waveforms, times = loaded_strain
        waveforms = np.asarray(waveforms, dtype=np.float32)
        times = np.asarray(times, dtype=np.float32)
    else:
        arr = np.asarray(loaded_strain)
        if arr.ndim == 3 and arr.shape[2] >= 2:
            waveforms = arr[:, :, 0].astype(np.float32)
            times = arr[:, :, 1].astype(np.float32)
        elif arr.ndim == 2:
            waveforms = arr.astype(np.float32)
            if time_axes is not None:
                times = np.asarray(time_axes, dtype=np.float32)
            else:
                # approximate time axis per simulation using start_time_vec and uniform spacing
                T = waveforms.shape[1]
                times = np.zeros_like(waveforms, dtype=np.float32)
                for i in range(n_sims):
                    # fallback: linspace from start_time to start_time + T-1
                    times[i, :] = np.linspace(start_time_vec[i], start_time_vec[i] + (T - 1), T, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported loaded_strain shape: {arr.shape}")

    if waveforms.shape[0] != params_np.shape[0]:
        raise ValueError("Number of waveforms does not match number of parameter rows in simulations_df")

    n, T = waveforms.shape
    l_p = params_np.shape[1]

    # Vectorized construction: repeat params, flatten times and waveforms
    # params_rep: (n*T, l_p)
    params_rep = np.repeat(params_np, T, axis=0).astype(np.float32)
    times_flat = times.reshape(-1).astype(np.float32)
    inputs = np.concatenate([params_rep, times_flat[:, None]], axis=1)
    targets = waveforms.reshape(-1).astype(np.float32)

    if save_npz is not None:
        np.savez_compressed(save_npz, inputs=inputs, targets=targets, params=params_np, time_axes=times)
        print(f"Saved processed arrays to {save_npz}")

    return inputs, targets, params_np, times


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SIREN from preprocessed arrays (npz).')
    parser.add_argument('--data', type=str, help='Path to processed .npz containing inputs and targets', required=False)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--checkpoint', type=str, default='siren_checkpoint.pt')
    args = parser.parse_args()

    if args.data is None:
        print('No --data provided. Exiting. Use this script by passing a processed .npz with arrays "inputs" and "targets".')
        sys.exit(1)
    else:
        p = Path(args.data)
        if not p.exists():
            print(f'File not found: {p}')
            sys.exit(1)
        else:
            d = np.load(str(p))
            inputs = d['inputs']
            targets = d['targets']
            in_features = inputs.shape[1]

    simulations_df, start_time_vec = get_sxs_simulation_list(100, 'noneccentric')
    sxs_id_list = list(map(str, simulations_df.index))

    start_time = time.time()
    loaded_strain = load_sxs_data(sxs_id_list, start_time_vec, time_axis_size=500)
    end_time = time.time()
    print(f"Loaded SXS data in {end_time - start_time:.2f} seconds.")

    inputs, targets, params_np, times = build_inputs_targets_from_loaded(
        simulations_df, start_time_vec, loaded_strain, save_npz='processed.npz')
    model = train_siren_arrays(
        inputs, targets, in_features=in_features, hidden_features=args.hidden, hidden_layers=args.layers,
        epochs=args.epochs, batch_size=args.bs, lr=args.lr, checkpoint_path=args.checkpoint)




