#!/usr/bin/env python3
"""
Author: Ritik Shah

User-configurable script for SpectraLift execution.
"""

# === User parameters ===
synthetic = True                           # Keep True if you wish to generate synthetic LR HSI and HR MSI, else change to False
provide_psf = False                         # set to True to load your own PSF.mat
provide_srf = False                         # set to True to load your own SRF.mat. If you want to run SpectraLift on a real world dataset (non synthetic)
                                            # and the HR input does not have 1,3,4,8, or 16 bands you must set this to True and provide a path to 
                                            # the srf file (you can give Gaussian estimates of the SRF)

psf_file = "/path_to_your_psf.mat"           # You can keep the psf_type, sigma, and kernel size as their default values if you have set provide_psf to True
srf_file = "/path_to_your_srf.mat"           # You can keep the num_msi_bands, fwhm_factor as their default values if you have set provide_srf to True

# For synthetic generation:
mat_file        = "/path_to_your_hsi.mat"    # Path to your raw HSI .mat file if you wish to generate synthetic inputs
mat_key         = "hsi_key"                     # Key for the raw HSI within your .mat file, this is case sensitive so please ensure you put in the exact key

psf_type        = "gaussian"               # This is case sensitive and the PSF types supported by our code are given in the next line. Please put the exact option given below
                                           # Options: 'gaussian', 'kolmogorov', 'airy', 'moffat', 'sinc', 'lorentzian2', 'hermite', 'parabolic', 'gabor', 'delta' 
sigma           = 3.4                      # Sigma defines the spread or blur scale of each PSF, controlling its sharpness or dispersion.
kernel_size     = 7                        # PSF kernel will be generated of shape (2*kernel_size + 1, 2*kernel_size + 1)

downsample_ratio = 4                          # Spatial downsampling factor
snr_spatial     = 35.0                        # SNR (dB) for LR HSI

num_msi_bands   = 4                           # Number of bands for HR MSI
snr_spectral    = 40.0                        # SNR (dB) for HR MSI
fwhm_factor     = 4.2                         # FWHM→σ factor for spectral degradation (this is for Gaussian approximation of the SRFs)

seed            = 42                          # Random seed

# For user-provided data (when synthetic=False):
hr_msi_file     = "/path_to_your_hr_msi.mat"        # Path to the LR HSI .mat file (please ensure this has only one key) or as a .tif file.
lr_hsi_file     = "/path_to_your_lr_hsi.mat"        # Path to the LR HSI .mat file (please ensure this has only one key) or as a .tif file.
num_bands_msi_non_synthetic = 3            # If you want the framework to generate a Gaussian estimate of an SRF for you, change this to reflect... 
                                              # ...the number of bands you want in your HR input. Options: 1,3,4,8,16...
                                              # ...If you want an SRF for a different number of bands, you must keep this None and provide srf file path above

# For SpectraLift execution:
verbose = 1                                  # 0: no information about the inputs or ouputs will be shown, 1: shapes of inputs and outputs will be shown
lr_schedule  = "cosine_restart"                   # This is case sensitive. Options: "one_cycle", "cosine_restart", or "flat"
init_lr      = 1e-4
max_lr       = 1e-2
final_lr     = 1e-6    # only for one_cycle
min_lr       = 1e-6    # only for cosine_restart
num_restarts = 2       # how many restarts in cosine schedule
epochs = 2500          # Number of training epochs
hidden_size = 64       # Number of neurons in each hidden layer of SIN
training_batch_size = None # Keep this none if you wish to train on full image, else specify an integer size and the training will happen on training_batch_size*training_batch_size number of pixels
inference_batch_size = 256 # Keep this none if you wish to infer on full image, else specify an integer size and the inference will happen on inference_batch_size*inference_batch_size number of pixels

output_file_type = 'h5'                   # Enter 'numpy' if you want the output to be saved as a .npy file, 'h5' if you want the output to be stored as a h5py file,
                                              # or 'matlab' if you want the output to be saved as a .mat file. This is case sensitive
# =========================

from generate_synthetic_inputs import generate_synthetic_inputs
from compute_metrics import show_evaluation_hsi
from spectralift_helpers import run_pipeline
from utils import normalize, normalize_srf, get_srf_bands, apply_srf
from tensorflow.keras import backend as K
import scipy.io as sio
import h5py
import gc
import os
from PIL import Image
import numpy as np

def load_single_var(mat_path):
    """
    Load a .mat file and return its single non-metadata variable.
    """
    mat = sio.loadmat(mat_path)
    data_keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(data_keys) != 1:
        raise KeyError(f"Expected one variable in {mat_path}, found {data_keys}")
    return mat[data_keys[0]]

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

# === Data acquisition ===
if synthetic and not provide_psf and not provide_srf:
    
    gt, hr_msi, lr_hsi, srf = generate_synthetic_inputs(
        mat_file=mat_file,
        mat_key=mat_key,
        psf_type=psf_type,
        sigma=sigma,
        kernel_size=kernel_size,
        downsample_ratio=downsample_ratio,
        snr_spatial=snr_spatial,
        num_msi_bands=num_msi_bands,
        snr_spectral=snr_spectral,
        fwhm_factor=fwhm_factor,
        true_psf=None,
        true_srf=None,
        seed=seed
    )

elif synthetic and provide_psf and not provide_srf:
    user_psf = load_single_var(psf_file)
    user_psf = user_psf / user_psf.sum()
    if user_psf.ndim != 2:
        raise ValueError("Loaded PSF must be 2D since it is applied per channel")
    elif user_psf.shape[0] != user_psf.shape[1]:
        raise ValueError("PSF must have a square kernel (shape of 1st dimension must equal shape of second dimension)")

    gt, hr_msi, lr_hsi, srf = generate_synthetic_inputs(
        mat_file=mat_file,
        mat_key=mat_key,
        psf_type=psf_type,
        sigma=sigma,
        kernel_size=kernel_size,
        downsample_ratio=downsample_ratio,
        snr_spatial=snr_spatial,
        num_msi_bands=num_msi_bands,
        snr_spectral=snr_spectral,
        fwhm_factor=fwhm_factor,
        true_psf=user_psf,
        true_srf=None,
        seed=seed
    )

elif synthetic and not provide_psf and provide_srf:
    srf = load_single_var(srf_file)
    if srf.ndim != 2:
        raise ValueError("Loaded SRF must be 2-D")
    # if the first dimension is smaller than the second, it's probably loaded (hsi,msi) and needs transposing
    if srf.shape[0] > srf.shape[1]:
        srf = np.transpose(srf)

    # Normalizing the SRF
    for i in range(srf.shape[0]):  # 0 axis is the band
        srf[i, :] = normalize_srf(srf[i, :])

    gt, hr_msi, lr_hsi, srf = generate_synthetic_inputs(
        mat_file=mat_file,
        mat_key=mat_key,
        psf_type=psf_type,
        sigma=sigma,
        kernel_size=kernel_size,
        downsample_ratio=downsample_ratio,
        snr_spatial=snr_spatial,
        num_msi_bands=num_msi_bands,
        snr_spectral=snr_spectral,
        fwhm_factor=fwhm_factor,
        true_psf=None,
        true_srf=srf,
        seed=seed
    )

elif synthetic and provide_psf and provide_srf:
    user_psf = load_single_var(psf_file)
    user_psf = user_psf / user_psf.sum()
    if user_psf.ndim != 2:
        raise ValueError("Loaded PSF must be 2D since it is applied per channel")
    elif user_psf.shape[0] != user_psf.shape[1]:
        raise ValueError("PSF must have a square kernel (shape of 1st dimension must equal shape of second dimension)")
    srf = load_single_var(srf_file)
    if srf.ndim != 2:
        raise ValueError("Loaded SRF must be 2-D")
    # if the first dimension is smaller than the second, it's probably loaded (hsi,msi) and needs transposing
    if srf.shape[0] > srf.shape[1]:
        srf = np.transpose(srf)

    # Normalizing the SRF
    for i in range(srf.shape[0]):  # 0 axis is the band
        srf[i, :] = normalize_srf(srf[i, :])

    gt, hr_msi, lr_hsi, srf = generate_synthetic_inputs(
        mat_file=mat_file,
        mat_key=mat_key,
        psf_type=psf_type,
        sigma=sigma,
        kernel_size=kernel_size,
        downsample_ratio=downsample_ratio,
        snr_spatial=snr_spatial,
        num_msi_bands=num_msi_bands,
        snr_spectral=snr_spectral,
        fwhm_factor=fwhm_factor,
        true_psf=user_psf,
        true_srf=srf,
        seed=seed
    )

elif not synthetic and provide_srf:
    print("Loading user-provided MSI/HSI/SRF from .mat files…")
    hr_msi = normalize(load_multiband(hr_msi_file))
    lr_hsi = normalize(load_multiband(lr_hsi_file))
    gt = None
    srf    = load_single_var(srf_file)
    if srf.ndim != 2:
        raise ValueError("Loaded SRF must be 2-D")
    # if the first dimension is smaller than the second, it's probably loaded (hsi,msi) and needs transposing
    if srf.shape[0] > srf.shape[1]:
        srf = np.transpose(srf)

    # Normalizing the SRF
    for i in range(srf.shape[0]):  # 0 axis is the band
        srf[i, :] = normalize_srf(srf[i, :])

else:
    print("Loading user-provided MSI/HSI/SRF from .mat files…")
    hr_msi = normalize(load_multiband(hr_msi_file))
    lr_hsi = normalize(load_multiband(lr_hsi_file))
    gt = None
    valid_band_counts = [1, 3, 4, 8, 16]
    if num_bands_msi_non_synthetic in valid_band_counts:
        band_specs = get_srf_bands(num_bands_msi_non_synthetic)
        _, srf, _ = apply_srf(lr_hsi, band_specs, fwhm_factor)
    else:
        raise ValueError(
            "We cannot generate an SRF for your chosen number of bands."
            "Please set provide_srf to True and give the path to your srf.mat file in the user parameters"
        )

# === Verbose output ===
if verbose >= 1:
    if gt is not None:
        print(f"GT HSI:   {gt.shape}")
    print(f"HR MSI:   {hr_msi.shape}")
    print(f"LR HSI:   {lr_hsi.shape}")
    print(f"SRF:      {srf.shape}")

# Get the directory where run.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output subfolder path
output_dir = os.path.join(current_dir, "Super Resolved Outputs")

# Create the subfolder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created folder: {output_dir}")
else:
    print(f"Folder already exists: {output_dir}")

# Run the SpectraLift pipeline on the inputs
SR_image = run_pipeline(
    hr_msi, lr_hsi, srf,
    lr_schedule=lr_schedule,
    initial_lr=init_lr,
    max_lr=max_lr,
    final_lr=final_lr,
    min_lr=min_lr,
    num_restarts=num_restarts,
    sin_hidden_size=hidden_size,
    num_epochs=epochs,
    training_batch_size=training_batch_size,
    inference_batch_size=inference_batch_size,
    synthetic=synthetic
)

if verbose >= 1:
    print(f"Super resolved image:   {SR_image.shape}")

# Saving the output to the folder 'Super Resolved Outputs'
# Build base filename
if synthetic:
    base_name = f"super_resolved_hsi_{mat_key}_{psf_type}_{downsample_ratio}_{num_msi_bands}"
else:
    base_name = "super_resolved_hsi"

if output_file_type == 'numpy':
    out_path = os.path.join(output_dir, base_name + '.npy')
    np.save(out_path, SR_image)
elif output_file_type == 'matlab':
    out_path = os.path.join(output_dir, base_name + '.mat')
    sio.savemat(out_path, {base_name: SR_image})
elif output_file_type == 'h5':
    out_path = os.path.join(output_dir, base_name + '.h5')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset(base_name, data=SR_image, compression='gzip', chunks=True)
else:
    raise ValueError("output_file_type must be 'numpy', 'h5' or 'matlab'")

print(f"Saved super-resolved image → {out_path}")

# Calculating and printing metrics if synthetic data was used
if synthetic:
    show_evaluation_hsi(gt, SR_image)

# Freeing up GPU memory
try:
    del model
    del optimizer
except NameError:
    pass

K.clear_session()
gc.collect()