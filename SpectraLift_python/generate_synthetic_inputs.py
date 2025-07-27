# Author: Ritik Shah

import numpy as np
import scipy.io as sio

from utils import (
    set_random_seed,
    normalize,
    spectral_degradation,
    spatial_degradation,
    gaussian_psf,
    kolmogorov_psf,
    airy_psf,
    moffat_psf,
    sinc_psf,
    lorentzian_squared_psf,
    hermite_psf,
    parabolic_psf,
    gabor_psf,
    delta_function_psf
)

# Map string keys to your PSF constructors
PSF_FUNCS = {
    "gaussian":          gaussian_psf,
    "kolmogorov":        kolmogorov_psf,
    "airy":              airy_psf,
    "moffat":            moffat_psf,
    "sinc":              sinc_psf,
    "lorentzian2":       lorentzian_squared_psf,
    "hermite":           hermite_psf,
    "parabolic":         parabolic_psf,
    "gabor":             gabor_psf,
    "delta":             delta_function_psf,
}

def generate_synthetic_inputs(
    mat_file: str,
    mat_key: str            = "dc",
    psf_type: str           = "gaussian",
    sigma: float            = 3.40,
    kernel_size: int        = 7,
    downsample_ratio: int   = 4,
    snr_spatial: float      = 35.0,
    num_msi_bands: int      = 4,
    snr_spectral: float     = 40.0,
    fwhm_factor: float      = 4.2,
    true_psf: np.ndarray    = None,
    true_srf: np.ndarray    = None,
    seed: int               = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # 1) reproducibility
    set_random_seed(seed)

    # 2) load raw HSI
    mat = sio.loadmat(mat_file)
    if mat_key not in mat:
        raise KeyError(f"Key '{mat_key}' not found in {mat_file}")
    raw_image = mat[mat_key].astype(np.float32)
    gt = normalize(raw_image)

    # 3) spectral degradation → HR MSI + SRF
    hr_msi, srf, _, _ = spectral_degradation(  # change the logic slightly to allow the user to give an SRF for synthetic generation
        image=raw_image,
        SNR=snr_spectral,
        num_bands=num_msi_bands,
        fwhm_factor=fwhm_factor,
        user_srf=true_srf
    )

    # 4) build PSF & spatial degradation → LR HSI
    if true_psf is None:
        psf_fn = PSF_FUNCS.get(psf_type.lower())
        if psf_fn is None:
            raise ValueError(f"Unknown psf_type '{psf_type}'; choose one of {list(PSF_FUNCS)}")
        psf = psf_fn(sigma, kernel_size)
    else:
        psf = true_psf

    lr_hsi = spatial_degradation(
        image=raw_image,
        psf=psf,
        downsample_ratio=downsample_ratio,
        SNR=snr_spatial
    )

    return gt, hr_msi, lr_hsi, srf