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
from torch.cuda.amp import autocast, GradScaler
from SIREN import Siren

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

def load_sxs_data(sxs_id_list: int, start_time_vec: int, time_axis_size: int = 1000, dom_em: int  = 2, dom_ell: int = 2):
    waveforms_array = np.empty((len(sxs_id_list), time_axis_size, 2), dtype=np.float32)
    for idx, sxs_id_str in enumerate(sxs_id_list):
        
        print(f"Loading simulation: {sxs_id_str}")
        try:
            simulation = sxs.load(sxs_id_str, download=True, progress=True,
                                # ignore_deprecation=True,
                                # auto_supersede=True # I learned I've been misspelling this all my life
                                )
        except Exception as e: print(f"Error loading simulation {sxs_id_str}: {e}"); raise
        
        strain_modes_obj = getattr(simulation, 'h', None)
        if strain_modes_obj is None:
            raise ValueError(f"Strain not found or empty for simulation {sxs_id_str}.")

        peak_strain_time = strain_modes_obj.max_norm_time()

        time_axis = np.linspace(start_time_vec[idx], peak_strain_time, time_axis_size)
        # Extract just the dominant mode
        dom_strain_signal = strain_modes_obj[:, strain_modes_obj.index(dom_ell, dom_em)].real
        # ^ Using real part; could also use imag or abs (for non-periodic amplitude)
        
        interpolated_waveform = dom_strain_signal.interpolate(time_axis)
        waveforms_array[idx, :, 0] = interpolated_waveform.ndarray.real
        waveforms_array[idx, :, 1] = time_axis
        
    print(waveforms_array)

    return dom_strain_signal

simulations_df, start_time_vec = get_sxs_simulation_list(100, 'noneccentric')
print(simulations_df.head(), simulations_df.shape, '\n', start_time_vec)

sxs_id_list = list(map(str, simulations_df.index))

start_time = time.time()
loaded_strain = load_sxs_data(sxs_id_list, start_time_vec, time_axis_size=500)
end_time = time.time()
print(f"Loaded SXS data in {end_time - start_time:.2f} seconds.")
print(loaded_strain.shape)