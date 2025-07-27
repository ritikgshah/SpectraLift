"""
Author: Ritik Shah
Spectralift Package Initialization

Expose core functions for easy import when using Spectralift as a module.
"""

# Import core utilities
from .utils import (
    set_random_seed,
    normalize,
    normalize_srf,
    gaussian_psf,
    kolmogorov_psf,
    airy_psf,
    moffat_psf,
    sinc_psf,
    lorentzian_squared_psf,
    hermite_psf,
    parabolic_psf,
    gabor_psf,
    delta_function_psf,
    spectral_degradation,
    spatial_degradation,
)

# Synthetic input generation
from .generate_synthetic_inputs import (
    generate_synthetic_inputs,
    PSF_FUNCS,
)

# SpectraLift pipeline and helpers
from .spectralift_helpers import (
    run_pipeline,
    SpectralSR_MLP,
    train_spectral_mlp,
    infer_and_analyze_model_performance_tf,
)

# Metric computation
from .compute_metrics import (
    show_evaluation_hsi
)

__all__ = [
    # utils
    'set_random_seed', 'normalize',
    'gaussian_psf', 'kolmogorov_psf', 'airy_psf', 'moffat_psf',
    'sinc_psf', 'lorentzian_squared_psf', 'hermite_psf',
    'parabolic_psf', 'gabor_psf', 'delta_function_psf',
    'spectral_degradation', 'spatial_degradation',
    # synthetic inputs
    'generate_synthetic_inputs', 'PSF_FUNCS',
    # pipeline
    'run_pipeline', 'SpectralSR_MLP', 'train_spectral_mlp',
    'infer_and_analyze_model_performance_tf',
]