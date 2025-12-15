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
from SHAN import SHAN # custom SHAN build
import argparse
from pathlib import Path

def get_sxs_simulation_list(num_simulations, eccentricity: str = 'noneccentric') -> pd.DataFrame:
    df = sxs.load("dataframe", tag="3.0.0")
    df_good = getattr(df.undeprecated.BBH, eccentricity)
    if df_good.shape[0] < num_simulations:
        print(f"Warning: Requested {num_simulations} simulations but only {df_good.shape[0]} available.")
    elif df_good.shape[0] > num_simulations:
        print(f"Selecting {num_simulations} simulations randomly from {df_good.shape[0]} available.")
        df_good = df_good.sample(n=num_simulations, random_state=43)
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

def load_sxs_data(sxs_id_list: int, start_time_vec: int, time_axis_size: int = 1000, dom_em: int  = 2, dom_ell: int = 2):
    """Load and interpolate SXS dominant mode strain for a list of simulation ids.

    Returns:
      waveforms: np.ndarray shape (n_sims, time_axis_size) -- real strain values
      time_axes: np.ndarray shape (n_sims, time_axis_size) -- time axes used for each sim
    """
    n = len(sxs_id_list)
    waveforms = np.zeros((n, time_axis_size), dtype=np.float32)
    time_axes = np.zeros((n, time_axis_size), dtype=np.float32)

    for idx, sxs_id_str in enumerate(sxs_id_list):
        print(f"Loading simulation: {sxs_id_str} ({idx+1}/{n})")
        # load simulation

        try:
            simulation = sxs.load(sxs_id_str, download=True, progress=False)
        except Exception as e:
            print(f"Error loading simulation {sxs_id_str}: {e}")
            raise

        strain_modes_obj = getattr(simulation, 'h', None)
        if strain_modes_obj is None:
            raise ValueError(f"Strain not found or empty for simulation {sxs_id_str}.")

        peak_strain_time = strain_modes_obj.max_norm_time()
        # Extract dominant (ell, m) mode and take real part
        dom_mode = strain_modes_obj[:, strain_modes_obj.index(dom_ell, dom_em)]
        # Make the wave start at a local peak for consistency
        settled_idx = dom_mode.index_closest_to(start_time_vec[idx])
        array_for_finding_peak = np.abs(dom_mode.ndarray.real[settled_idx:])
        peak_start_idx = scipy.signal.argrelextrema(array_for_finding_peak, np.less)[0][0]
        true_start_time = dom_mode.time[settled_idx + peak_start_idx]
        
        time_axis = np.linspace(true_start_time, peak_strain_time, time_axis_size)
        interpolated = dom_mode.interpolate(time_axis)
        wf = np.asarray(interpolated.ndarray.real, dtype=np.float32)
        waveforms[idx, :] = wf
        time_axes[idx, :] = np.linspace(0, 1000, time_axis_size, dtype=np.float32)  # normalized time axis


    return waveforms, time_axes


def train_siren_arrays(inputs: np.ndarray,
                       targets: np.ndarray,
                       in_features: int,
                       hidden_features: int = 256,
                       hidden_layers: int = 3,
                       epochs: int = 20,
                       batch_size: int = 4096,
                       lr: float = 1e-4,
                       device: str = None,
                       checkpoint_path: str = 'siren_checkpoint.pt',
                       val_split: float = 0.1,
                       plot_path: str = 'training_loss.png',
                       weight_decay: float = 1e-6,
                       clip_grad_norm: float = 1.0,
                       lr_patience: int = 10,
                       min_lr: float = 1e-7) -> SHAN:
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

    # prepare tensors and split into train/val
    N = inputs.shape[0]
    tensor_x = torch.from_numpy(inputs.astype(np.float32))
    tensor_y = torch.from_numpy(targets.astype(np.float32))

    # split indices
    indices = np.arange(N)
    if val_split and 0.0 < val_split < 1.0:
        np.random.shuffle(indices)
        n_val = int(N * val_split)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
    else:
        train_idx = indices
        val_idx = np.array([], dtype=int)

    train_ds = TensorDataset(tensor_x[train_idx], tensor_y[train_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    if val_idx.size:
        val_ds = TensorDataset(tensor_x[val_idx], tensor_y[val_idx])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        val_loader = None

    model = SHAN(in_features=in_features, hidden_features=hidden_features,
                 hidden_layers=hidden_layers, enforce_positive_B=True).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=lr_patience, min_lr=min_lr)
    # loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    scaler = GradScaler()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            with autocast(device_type=device, enabled=(device != 'cpu')):
                preds = model(xb)
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            # unscale then clip if requested
            if clip_grad_norm is not None:
                try:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                except Exception:
                    pass
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            batches += 1

        avg_train = running / max(1, batches)
        train_losses.append(avg_train)

        # validation
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    with autocast(device_type=device, enabled=(device != 'cpu')):
                        preds = model(xb)
                        loss = loss_fn(preds, yb)
                    val_running += loss.item()
                    val_batches += 1
            avg_val = val_running / max(1, val_batches)
            val_losses.append(avg_val)
            print(f"Epoch {epoch}/{epochs}  train_loss={avg_train:.6e}  val_loss={avg_val:.6e}")
            # scheduler.step(avg_val)
        else:
            print(f"Epoch {epoch}/{epochs}  train_loss={avg_train:.6e}")
            # scheduler.step(avg_train)

        # checkpoint
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': opt.state_dict()}, checkpoint_path)

    print(f"Finished training, final checkpoint saved to {checkpoint_path}")

    # plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label='train')
        if val_losses:
            plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Saved training loss plot to {plot_path}")
    except Exception as e:
        print(f"Could not save plot (matplotlib missing or error): {e}")

    history = {'train_losses': train_losses, 'val_losses': val_losses}
    return model, history


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
    parser = argparse.ArgumentParser(description='Train SIREN from SXS data or a preprocessed .npz.')
    parser.add_argument('--data', type=str, help='Path to processed .npz containing inputs and targets (if provided, skips SXS loading)', required=False)
    parser.add_argument('--num-sims', type=int, default=100, help='Number of SXS simulations to select')
    parser.add_argument('--eccentricity', type=str, default='noneccentric', help='Which SXS subset to use')
    parser.add_argument('--time-size', type=int, default=500, help='Number of time samples per waveform')
    parser.add_argument('--save-processed', type=str, default='processed.npz', help='Where to save processed inputs/targets (.npz)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--checkpoint', type=str, default='shan_checkpoint.pt')
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--plot', type=str, default='shan_training_loss.png', help='Path to save training loss plot')
    args = parser.parse_args()

    if args.data:
        p = Path(args.data)
        if not p.exists():
            print(f'File not found: {p}')
            sys.exit(1)
        d = np.load(str(p))
        inputs = d['inputs']
        targets = d['targets']
        in_features = inputs.shape[1]
    else:
        # load SXS list and waveforms (with caching)
        simulations_df, start_time_vec = get_sxs_simulation_list(args.num_sims, args.eccentricity)
        sxs_id_list = list(map(str, simulations_df.index))

        start_time = time.time()
        waveforms, times = load_sxs_data(sxs_id_list, start_time_vec, time_axis_size=args.time_size)
        end_time = time.time()
        print(f"Loaded SXS data in {end_time - start_time:.2f} seconds.")

        # If the user passed just a filename, place it under
        # 'processed_data/' folder. Creates directories as needed
        if args.save_processed:
            sp = Path(args.save_processed)
            if sp.parent == Path('.'):
                sp = Path('processed_data') / sp.name
            sp.parent.mkdir(parents=True, exist_ok=True)
            save_processed_path = str(sp)
        else:
            save_processed_path = None

        inputs, targets, params_np, times = build_inputs_targets_from_loaded(
            simulations_df, start_time_vec, (waveforms, times), save_npz=save_processed_path)
        in_features = inputs.shape[1]

    # Ensure checkpoint directory exists, adds to 'checkpoint_models/' folder if needed
    cp = Path(args.checkpoint)
    if cp.parent == Path('.'):
        cp = Path('checkpoint_models') / cp.name
    cp.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(cp)

    # Same for the plot path; 'training_plots/' folder
    plot_path = Path(args.plot)
    if plot_path.parent == Path('.'):
        plot_path = Path('training_plots') / plot_path.name
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path = str(plot_path)

    model, history = train_siren_arrays(
        inputs, targets, in_features=in_features, hidden_features=args.hidden, hidden_layers=args.layers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, checkpoint_path=checkpoint_path, val_split=args.val_split, plot_path=plot_path)




