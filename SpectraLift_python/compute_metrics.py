# Author: Ritik Shah

from sewar.full_ref import mse, rmse, psnr, ssim, msssim, uqi, vifp, ergas, scc, rase, sam
import numpy as np
import matplotlib.pyplot as plt
from spectral import *

def compute_mse(image1, image2):
    """
    Compute the Mean Squared Error (MSE) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - mse_value (float): The MSE value between the two images.
    """
    mse_value = mse(image1, image2)
    return mse_value

def compute_rmse(image1, image2):
    """
    Compute the Root Mean Squared Error (RMSE) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - rmse_value (float): The RMSE value between the two images.
    """
    rmse_value = rmse(image1, image2)
    return rmse_value

def compute_psnr(image1, image2):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - psnr_value (float): The PSNR value between the two images.
    """
    # Convert images to uint8 if they are in a floating-point format
    if np.issubdtype(image1.dtype, np.floating):
        image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
    if np.issubdtype(image2.dtype, np.floating):
        image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)

    # Compute PSNR
    psnr_value = psnr(image1, image2)
    return psnr_value

def compute_ssim(image1, image2):
    """
    Compute the Structural Similarity Index (SSIM) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - ssim_value (float): The SSIM value between the two images.
    """
    # Convert images to uint8 if they are in a floating-point format
    if np.issubdtype(image1.dtype, np.floating):
        image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
    if np.issubdtype(image2.dtype, np.floating):
        image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)

    # Compute SSIM
    ssim_value, _ = ssim(image1, image2)
    return ssim_value

def compute_ms_ssim(image1, image2):
    """
    Compute the Multi-scale Structural Similarity Index (MS-SSIM) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - ms_ssim_value (float): The MS-SSIM value between the two images.
    """
    # Convert images to uint8 if they are in a floating-point format
    if np.issubdtype(image1.dtype, np.floating):
        image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
    if np.issubdtype(image2.dtype, np.floating):
        image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)

    # Compute MS-SSIM
    ms_ssim_value = msssim(image1, image2)
    return ms_ssim_value

def compute_uiqi(image1, image2):
    """
    Compute the Universal Image Quality Index (UIQI) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - uiqi_value (float): The UIQI value between the two images.
    """
    uiqi_value = uqi(image1, image2)
    return uiqi_value

def compute_vif(image1, image2):
    """
    Compute the Visual Information Fidelity (VIF) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - vif_value (float): The VIF value between the two images.
    """
    vif_value = vifp(image1, image2)
    return vif_value

def compute_fsim(image1, image2):
    """
    Compute the Feature Similarity Index (FSIM) between two images using image-similarity-measures.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - fsim_value (float): The FSIM value between the two images.
    """
    # Ensure images are in the correct format for FSIM
    if image1.dtype != np.uint8:
        image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)
    
    # Compute FSIM
    fsim_value = fsim(image1, image2)
    return fsim_value

def compute_sre(image1, image2):
    """
    Compute the Signal to Reconstruction Error Ratio (SRE) between two images.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - sre_value (float): The SRE value between the two images.
    """
    # Ensure images are in the correct format for SRE
    if image1.dtype != np.uint8:
        image1 = np.clip(image1 * 255, 0, 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = np.clip(image2 * 255, 0, 255).astype(np.uint8)
    
    # Compute SRE
    sre_value = sre(image1, image2)
    return sre_value

def compute_ergas(image1, image2):
    """
    Compute the ERGAS between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - ergas_value (float): The ERGAS value between the two images.
    """
    ergas_value = ergas(image1, image2)
    return ergas_value

def compute_scc(image1, image2):
    """
    Compute the Spatial Correlation Coefficient (SCC) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - scc_value (float): The SCC value between the two images.
    """
    scc_value = scc(image1, image2)
    return scc_value

def compute_rase(image1, image2):
    """
    Compute the Relative Average Spectral Error (RASE) between two images using sewar.

    Args:
    - image1 (numpy.ndarray): The first image. Shape should be (height, width, channels).
    - image2 (numpy.ndarray): The second image. Shape should be the same as image1.

    Returns:
    - rase_value (float): The RASE value between the two images.
    """
    rase_value = rase(image1, image2)
    return rase_value

def compute_sam(label: np.ndarray, output: np.ndarray):
        h, w, c = label.shape
        x_norm = np.sqrt(np.sum(np.square(label), axis=-1))
        y_norm = np.sqrt(np.sum(np.square(output), axis=-1))
        xy_norm = np.multiply(x_norm, y_norm)
        xy = np.sum(np.multiply(label, output), axis=-1)
        dist = np.mean(np.arccos(np.minimum(np.divide(xy, xy_norm + 1e-8), 1.0 - 1.0e-9)))
        dist = np.multiply(180.0 / np.pi, dist)
        return dist
    
def evaluate_metrics_hsi(img1, img2):
    """
    Evaluate various image quality metrics between two hyperspectral images.

    Args:
    - img1 (numpy.ndarray): The first image (e.g., super resolved HSI).
    - img2 (numpy.ndarray): The second image (e.g., ground truth HSI).

    Returns:
    - metrics_dict (dict): Dictionary containing all the evaluated metrics.
    """
    # Ensure the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("The shapes of the two images must be the same.")

    # Calculate the metrics
    hsi_metrics_dict = {}
    hsi_metrics_dict['RMSE'] = compute_rmse(img1, img2)
    hsi_metrics_dict['PSNR'] = compute_psnr(img1, img2)
    hsi_metrics_dict['SSIM'] = compute_ssim(img1, img2)
    hsi_metrics_dict['UIQI'] = compute_uiqi(img1, img2)
    hsi_metrics_dict['ERGAS'] = compute_ergas(img1, img2)
    hsi_metrics_dict['SAM'] = compute_sam(img1, img2)

    return hsi_metrics_dict

def show_evaluation_hsi(gt, pred):
    
    # Evaluate metrics
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    
    hsi_metrics = evaluate_metrics_hsi(gt, pred)

    # Print the evaluated metrics for the current inference image
    print(f"Metrics comparing Super Resolved HSI to the ground truth:")
    for metric_name, value in hsi_metrics.items():
        print(f"{metric_name}: {value}")