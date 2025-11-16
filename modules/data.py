"""Utility functions for handling data."""

import mne
import numpy as np
from typing import List, Optional
from joblib import Parallel, delayed
from .utils import pad_nan_to_target


def gen_ano_train_data(all_train_data):
    maxl = np.max([len(all_train_data[k]) for k in all_train_data])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data


def load_fif(
    files: List[str],
    n_timestamps: int,
    step: Optional[int] = None,
    zscore: bool = True,
    n_jobs: int = 1,
    picks="meg",
    preload=True,
) -> np.ndarray:
    """
    Load .fif files (MEG) in parallel and return windows array shaped
    (n_instances, n_timestamps, n_features).

    Args:
        files:
            list of paths to .fif files.
        n_timestamps:
            number of timepoints per window (window length).
        step:
            step size between windows. If None, we use n_timestamps (no overlap).
        zscore:
            whether to z-score normalize per-channel across all returned windows.
        n_jobs:
            number of parallel workers for loading files (joblib.Parallel).
        picks:
            optional picks argument for mne.get_data.
        preload:
            whether to preload raw data when reading files (True recommended).

    Returns:
        train_data:
            np.ndarray, shape (n_instances, n_timestamps, n_features),
            dtype float32
    """
    if step is None:
        step = n_timestamps

    if isinstance(files, str):
        files = [files]

    def _load(path: str):
        # Helper function to load single fif file

        print("Loading", path)

        # Read raw
        raw = mne.io.read_raw_fif(path, preload=preload, verbose=False)
        data = raw.get_data(picks=picks)
        n_channels, n_times = data.shape

        if n_times < n_timestamps:
            # no full window available
            return np.empty((0, n_timestamps, n_channels), dtype=np.float32)

        # number of windows (drop tail incomplete window)
        n_windows = 1 + (n_times - n_timestamps) // step
        if n_windows <= 0:
            return np.empty((0, n_timestamps, n_channels), dtype=np.float32)

        windows = np.empty((n_windows, n_timestamps, n_channels), dtype=np.float32)
        start = 0
        for i in range(n_windows):
            s = start + i * step
            e = s + n_timestamps
            windows[i] = data[:, s:e].T  # transpose -> (n_timestamps, n_channels)

        return windows

    # load in parallel
    try:
        results = Parallel(n_jobs=n_jobs)(delayed(_load)(path) for path in files)
    except Exception as e:
        raise RuntimeError(f"Error loading .fif files in parallel: {e}")

    # concatenate windows from all files
    if len(results) == 0:
        return np.empty((0, n_timestamps, 0), dtype=np.float32)

    windows_list = [r for r in results if r.size > 0]
    if len(windows_list) == 0:
        # no full windows across all files
        return np.empty((0, n_timestamps, 0), dtype=np.float32)

    train_data = np.concatenate(
        windows_list, axis=0
    )  # (n_instances, n_timestamps, n_channels)
    train_data = train_data.astype(np.float32)

    if zscore and train_data.size > 0:
        feat_mean = train_data.mean(axis=(0, 1), keepdims=True)
        feat_std = train_data.std(axis=(0, 1), keepdims=True)
        train_data = (train_data - feat_mean) / feat_std

    return train_data


def load_npy(
    files: List[str],
    n_timestamps: int,
    step: Optional[int] = None,
    zscore: bool = True,
    time_axis_first: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Load .npy files and return windows array shaped
    (n_instances, n_timestamps, n_features).

    Each .npy file is assumed to contain a continuous recording:
        - shape (n_channels, n_times) OR
        - shape (n_times, n_channels)

    Args:
        files:
            list of paths to .npy files.
        n_timestamps:
            number of timepoints per window (window length).
        step:
            step size between windows. If None, we use n_timestamps (no overlap).
        zscore:
            whether to z-score normalize per-channel across all returned windows.
        time_axis_first:
            is the time dimension the first one in the array?
        n_jobs:
            number of parallel workers for loading files (joblib.Parallel).

    Returns:
        train_data:
            np.ndarray, shape (n_instances, n_timestamps, n_features),
            dtype float32
    """
    if step is None:
        step = n_timestamps

    if isinstance(files, str):
        files = [files]

    def _load(path: str):
        print("Loading", path)

        data = np.load(path)

        if data.ndim != 2:
            raise ValueError(f"{path} has shape {data.shape}, expected 2D array")

        # ensure shape = (n_channels, n_times)
        if time_axis_first:
            # transpose
            n_times, n_channels = data.shape
            data_ct = data.T
        else:
            n_channels, n_times = data.shape
            data_ct = data

        if n_times < n_timestamps:
            return np.empty((0, n_timestamps, n_channels), dtype=np.float32)

        n_windows = 1 + (n_times - n_timestamps) // step
        if n_windows <= 0:
            return np.empty((0, n_timestamps, n_channels), dtype=np.float32)

        windows = np.empty((n_windows, n_timestamps, n_channels), dtype=np.float32)

        for i in range(n_windows):
            s = i * step
            e = s + n_timestamps
            windows[i] = data_ct[:, s:e].T  # (T, C)

        return windows

    # load in parallel
    try:
        results = Parallel(n_jobs=n_jobs)(delayed(_load)(path) for path in files)
    except Exception as e:
        raise RuntimeError(f"Error loading .npy files in parallel: {e}")

    if len(results) == 0:
        return np.empty((0, n_timestamps, 0), dtype=np.float32)

    windows_list = [r for r in results if r.size > 0]
    if len(windows_list) == 0:
        return np.empty((0, n_timestamps, 0), dtype=np.float32)

    train_data = np.concatenate(windows_list, axis=0)
    train_data = train_data.astype(np.float32)

    if zscore and train_data.size > 0:
        feat_mean = train_data.mean(axis=(0, 1), keepdims=True)
        feat_std = train_data.std(axis=(0, 1), keepdims=True) + 1e-6
        train_data = (train_data - feat_mean) / feat_std

    return train_data
