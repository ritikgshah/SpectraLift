# Author: Ritik Shah

import argparse
import os
import json
import yaml
import logging
import gc
from pathlib import Path

import numpy as np
import scipy.io as sio
import h5py
from tensorflow.keras import backend as K
from PIL import Image

from generate_synthetic_inputs import generate_synthetic_inputs
from compute_metrics import show_evaluation_hsi
from spectralift_helpers import run_pipeline
from utils import normalize, normalize_srf, get_srf_bands, apply_srf

def load_single_var(mat_path):
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(keys) != 1:
        raise KeyError(f"Expected one variable in {mat_path}, found {keys}")
    return mat[keys[0]]

def load_multiband(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mat':
        mat = sio.loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if len(keys) != 1:
            raise KeyError(f"Expected exactly one variable in {path}, got {keys}")
        arr = mat[keys[0]]
    elif ext in ('.tif', '.tiff'):
        with Image.open(path) as img:
            # PIL will load each band as a frame
            bands = []
            for i in range(img.n_frames):
                img.seek(i)
                bands.append(np.array(img))
            arr = np.stack(bands, axis=-1)
            arr = np.squeeze(arr, axis=-1)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'.  Use .mat or .tif/.tiff")
    return arr.astype(np.float32)

def setup_logging(log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def load_config():
    parser = argparse.ArgumentParser(description="Run SpectraLift from config file")
    parser.add_argument('--config', required=True, type=str, help="Path to YAML or JSON config file")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f), config_path
        elif config_path.endswith('.json'):
            return json.load(f), config_path
        else:
            raise ValueError("Unsupported config format: use .yaml, .yml or .json")

def main():
    config, config_path = load_config()

    log_dir = Path(config.get("log_dir", "logs"))
    setup_logging(log_dir / "spectralift.log")
    logging.info("SpectraLift started")

    # Save a snapshot of config
    with open(log_dir / "used_config.yaml", 'w') as f:
        yaml.dump(config, f)

    synthetic = config.get("synthetic", True)
    provide_psf = config.get("provide_psf", False)
    provide_srf = config.get("provide_srf", False)
    gt = None

    if synthetic:
        logging.info("Running in synthetic mode")

        user_psf = load_single_var(config["psf_file"]) if provide_psf else None
        if user_psf is not None:
            user_psf = user_psf / user_psf.sum()
            if user_psf.ndim != 2 or user_psf.shape[0] != user_psf.shape[1]:
                raise ValueError("PSF must be a 2D square matrix.")

        if provide_srf:
            srf = load_single_var(config["srf_file"])
            if srf.ndim != 2:
                raise ValueError("SRF must be a 2D matrix.")
            if srf.shape[0] > srf.shape[1]:
                srf = srf.T
            srf = np.stack([normalize_srf(row) for row in srf])
        else:
            srf = None

        gt, hr_msi, lr_hsi, srf = generate_synthetic_inputs(
            mat_file=config["mat_file"],
            mat_key=config["mat_key"],
            psf_type=config["psf_type"],
            sigma=config["sigma"],
            kernel_size=config["kernel_size"],
            downsample_ratio=config["downsample_ratio"],
            snr_spatial=config["snr_spatial"],
            num_msi_bands=config["num_msi_bands"],
            snr_spectral=config["snr_spectral"],
            fwhm_factor=config["fwhm_factor"],
            true_psf=user_psf,
            true_srf=srf,
            seed=config.get("seed", 42)
        )
    else:
        logging.info("Running in real-world mode")
        hr_msi = normalize(load_multiband(config["hr_msi_file"]))
        lr_hsi = normalize(load_multiband(config["lr_hsi_file"]))

        if provide_srf:
            srf = load_single_var(config["srf_file"])
            if srf.ndim != 2:
                raise ValueError("SRF must be 2D.")
            if srf.shape[0] > srf.shape[1]:
                srf = srf.T
            srf = np.stack([normalize_srf(row) for row in srf])
        else:
            n_bands = config["num_bands_msi_non_synthetic"]
            if n_bands in [1, 3, 4, 8, 16]:
                band_specs = get_srf_bands(n_bands)
                _, srf, _ = apply_srf(lr_hsi, band_specs, config["fwhm_factor"])
            else:
                raise ValueError("Cannot generate SRF for custom band count without srf_file.")

    if config.get("verbose", 1):
        if gt is not None:
            logging.info(f"GT shape: {gt.shape}")
        logging.info(f"HR MSI shape: {hr_msi.shape}")
        logging.info(f"LR HSI shape: {lr_hsi.shape}")
        logging.info(f"SRF shape: {srf.shape}")

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    SR_image = run_pipeline(
        hr_msi, lr_hsi, srf,
        lr_schedule=config["lr_schedule"],
        initial_lr=config["init_lr"],
        max_lr=config["max_lr"],
        final_lr=config["final_lr"],
        min_lr=config["min_lr"],
        num_restarts=config["num_restarts"],
        sin_hidden_size=config["hidden_size"],
        num_epochs=config["epochs"],
        training_batch_size=config["training_batch_size"],
        inference_batch_size=config["inference_batch_size"],
        synthetic=synthetic
    )

    base_name = (
        f"super_resolved_hsi_{config['mat_key']}_{config['psf_type']}_{config['downsample_ratio']}_{config['num_msi_bands']}"
        if synthetic else "super_resolved_hsi"
    )
    out_type = config["output_file_type"]
    out_path = output_dir / f"{base_name}.{ 'npy' if out_type == 'numpy' else 'mat' if out_type == 'matlab' else 'h5' }"

    if out_type == 'numpy':
        np.save(out_path, SR_image)
    elif out_type == 'matlab':
        sio.savemat(out_path, {base_name: SR_image})
    elif out_type == 'h5':
        with h5py.File(out_path, 'w') as f:
            f.create_dataset(base_name, data=SR_image, compression='gzip', chunks=True)
    else:
        raise ValueError("output_file_type must be 'numpy', 'h5', or 'matlab'")

    logging.info(f"Saved result to {out_path}")

    if synthetic:
        show_evaluation_hsi(gt, SR_image)

    K.clear_session()
    gc.collect()

if __name__ == "__main__":
    main()