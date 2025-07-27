# Author: Ritik Shah

import scipy.io as sio
import numpy as np
from tqdm import tqdm
import os
import math
import time
import io as iot
import sys
import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def numpy_to_tf(np_array):
    
    """
    Converts a numpy array into a tensorflow tensor.
    """
    
    tf_tensor = tf.constant(np_array, dtype=tf.float32)
    return tf_tensor

def tf_to_numpy(tf_tensor):
    
    """
    Converts a tensorflow tensor into a numpy array.
    """
    
    np_array = tf_tensor.numpy()
    return np_array

def apply_srf_tf(hsi, srf):
    """
    Tensorflow based function to apply a SRF to an image

    Parameters:
        hsi (tf.tensor): The hyperspectral image to which the SRF should be applied of shape (h,w,C)
        srf (tf.tensor): The srf to apply to the image of shape (msi_bands, hsi_bands)

    Returns: msi (tf.tensor): The multispectral image resulting from the application of the srf to the hyperspectral image of shape (h,w,c)
    """
    # Transpose SRF to shape (L_hsi, num_bands)
    srf_t = tf.transpose(srf)  # [L_hsi, num_bands]

    # Tensordot over the last axis of `image` and first axis of `srf_t`
    # Resulting shape = image.shape[:-1] + (num_bands,)
    msi = tf.tensordot(hsi, srf_t, axes=[[-1], [0]])

    return msi

# Function to apply PSF to an image in tensorflow
def apply_psf_tf(image, psf):
    """
    Applies the PSF via depthwise convolution on each spectral band.
    
    Parameters:
        image: tf.Tensor of shape (B, H, W, C)
        psf: np.ndarray of shape (k, k)

    Returns:
        tf.Tensor of shape (B, H, W, C)
    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    assert image_tensor.shape.rank == 4, f"Expected 4D input, got {image_tensor.shape}"
    
    # Get number of channels
    C = image_tensor.shape[-1]  # Must be spectral channels, e.g., 191

    # Prepare PSF kernel
    psf_tensor = tf.convert_to_tensor(psf, dtype=tf.float32)
    psf_tensor = tf.reshape(psf_tensor, [*psf_tensor.shape, 1, 1])  # (k, k, 1, 1)
    psf_tensor = tf.tile(psf_tensor, [1, 1, C, 1])  # (k, k, C, 1)

    # Depthwise convolution
    blurred = tf.nn.depthwise_conv2d(
        input=image_tensor,
        filter=psf_tensor,
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    return blurred

def downsample_image_to_reference_tf(image, reference, method='bicubic'):
    """
    Resize `image` to match the spatial dimensions of `reference` using
    differentiable TensorFlow ops while handling different channel sizes.
    
    Parameters:
    - image: tf.Tensor of shape (H1, W1, C1)
    - reference: tf.Tensor of shape (H2, W2, C2)
    - method: str, one of {'bilinear', 'nearest', 'bicubic', 'area', ...}
    
    Returns:
    - tf.Tensor of shape (H2, W2, C1) (preserves image's channels)
    """
    # Remove batch dimension if present
    if len(image.shape) == 4:
        image = tf.squeeze(image, axis=0)

    if len(reference.shape) == 4:
        reference = tf.squeeze(reference, axis=0)

    # Get spatial dimensions from reference
    target_size = tf.shape(reference)[0:2]

    # Resize with anti-aliasing (preserving input channels)
    resized = tf.image.resize(image, size=target_size, method=method, antialias=True)

    return resized

def prepare_inputs(hr_msi, lr_hsi, srf):
    """
    Prepares the data for training.

    Parameters:
        hr_msi (np.ndarray): The high spatial resolution multispectral image of shape (H,W,c)
        lr_hsi (np.ndarray): The low spatial resolution hyperspectral image of shape (h,w,C)
        srf (np.ndarray): The spectral response function of the MSI sensor (can be approximated gaussians) of shape (msi_bands, hsi_bands)

    Returns:
        hr_msi (tf.Tensor): The high spatial resolution multispectral image of shape (H,W,c)
        lr_hsi (tf.Tensor): The low spatial resolution hyperspectral image of shape (h,w,C)
        lr_msi (tf.Tensor): The low spatial resolution multispectral image of shape (h,w,c)
    """

    hr_msi = numpy_to_tf(hr_msi)
    lr_hsi = numpy_to_tf(lr_hsi)
    srf = numpy_to_tf(srf)
    lr_msi = apply_srf_tf(lr_hsi, srf)

    return hr_msi, lr_hsi, lr_msi, srf

def get_gpu_memory_mb():
    """Returns current GPU memory usage (in MB) for GPU:0 using TensorFlow."""
    mem_info = tf.config.experimental.get_memory_info('GPU:0')
    return mem_info['current'] / (1024 ** 2)  # bytes → MB

def infer_and_analyze_model_performance_tf(model, sample_inputs):
    """
    Analyzes model complexity: FLOPs, parameters, inference time, and GPU memory usage.
    
    Parameters:
    - model (tf.keras.Model): The model to evaluate.
    - sample_inputs (list of tf.Tensor): List of input tensors matching the model's expected input.
    """
    # 1) Convert sample_inputs into a concrete function
    input_signature = [tf.TensorSpec(shape=inp.shape, dtype=inp.dtype) for inp in sample_inputs]

    # Properly trace the model using a callable
    @tf.function
    def model_fn(msi):
        return model(msi)

    concrete_func = model_fn.get_concrete_function(*input_signature)

    # 2) Freeze the graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    # 3) Compute FLOPs
    try:
        original_stdout = sys.stdout
        sys.stdout = iot.StringIO()

        with tf.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts["output"] = "none"
            flops = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=run_meta,
                options=opts
            ).total_float_ops
    finally:
        sys.stdout = original_stdout

    # 4) Count parameters and record starting GPU memory
    num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    start_mem = get_gpu_memory_mb()

    # 5) Time inference
    start = time.perf_counter()
    SR_image = model(*sample_inputs)
    end = time.perf_counter()
    inference_time = end - start

    # 6) GPU memory
    end_mem = get_gpu_memory_mb()
    mem_used = end_mem - start_mem

    return SR_image, num_params, flops, mem_used, inference_time

def batched_inference(model, hr_msi, batch_size, synthetic=False):
    if isinstance(hr_msi, tf.Tensor):
        hr_msi = hr_msi.numpy()
  
    H, W, C = hr_msi.shape   

    bs      = batch_size or max(H, W)
    ys      = list(range(0, H, bs))
    xs      = list(range(0, W, bs))
    total_batches = len(ys) * len(xs)

    # Build and freeze your graph ONE time.
    @tf.function
    def infer_fn(msi):
        return model(msi)
    concrete = infer_fn.get_concrete_function(
        tf.TensorSpec((None, None, C), tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete).graph.as_graph_def()

    # Now tile
    SR       = np.zeros((H, W, model.num_outputs), np.float32)
    pbar = tqdm(total=total_batches, desc="Inference")
    for y0 in ys:
        for x0 in xs:
            patch   = hr_msi[y0:y0+bs, x0:x0+bs]
            patch_tf = tf.constant(patch, tf.float32)
            sr_patch = model(patch_tf).numpy()

            SR[y0:y0+bs, x0:x0+bs] = sr_patch
            pbar.update(1)
    pbar.close()

    return SR

class SpectralSR_MLP(Model):
    def __init__(self, num_outputs, hidden_size=128):
        """
        Initializes the MLP model to spectrally super resolve a multispectral image into a hyperspectral image.

        Parameters:
            num_outputs (int): Number of output channels (C).
            hidden_size (int): Number of hidden units in the initial layers.
        """
        super(SpectralSR_MLP, self).__init__()
        self.num_outputs = num_outputs

        # Define the MLP layers
        self.layer1 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer2 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer3 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer4 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer5 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)
        self.layer6 = Dense(hidden_size, activation=LeakyReLU(alpha=0.3), dtype=tf.float32)

        self.output_layer = Dense(self.num_outputs, activation='linear', dtype=tf.float32)
        
    def call(self, msi):
        """
        Forward pass through the MLP.

        Parameters:
            msi (tf.Tensor): MSI of shape (H, W, c).

        Returns:
            hsi (tf.Tensor): MSI of shape (H, W, C).
        """
        
        # 1) grab dynamic sizes
        H = tf.shape(msi)[0]
        W = tf.shape(msi)[1]

        # 2) flatten for MLP
        x = msi
        x = tf.reshape(x, [H * W, -1])             

        # 3) pass through MLP
        x  = self.layer1(x)          # first projection to hidden_size
        x1 = x
        x  = self.layer2(x) + x1
        
        x2 = x
        x  = self.layer3(x)
        x  = self.layer4(x) + x2

        x3 = x
        x = self.layer5(x)
        x = self.layer6(x) + x3

        # 4) final MLP estimated hsi
        hsi = self.output_layer(x)                       # (*H*W, q)
        hsi = tf.reshape(hsi, [H, W, self.num_outputs])  # (H, W, q)
        return hsi

def train_spectral_mlp(
    lr_msi, lr_hsi,
    epochs: int = 2500,
    lr_schedule: str = "one_cycle",   # "one_cycle" | "cosine_restart" | "flat"
    init_lr: float = 1e-4,
    max_lr: float = 1e-2,
    final_lr: float = 1e-6,
    min_lr: float = 1e-6,              # only used for cosine_restart
    num_restarts: int = 1,             # only used for cosine_restart
    hidden_size: int = 64,
    batch_size=1024):                  
    """
    Trains the spectral super resolution MLP using the One-Cycle learning rate policy.

    Parameters:
        lr_msi (tf.Tensor): Low-res MSI (h, w, c)
        lr_hsi (tf.Tensor): Low-res HSI (h, w, C)
        epochs (int): Total number of epochs
        init_lr (float): Initial learning rate
        max_lr (float): Peak learning rate
        final_lr (float): Learning rate at end of training
        hidden_size (int): Number of hidden units in MLP
        batch_size: tile size for batching (B). If None, runs full image at once.

    Returns:
        trained_spectralsr_mlp (tf.keras.Model): Trained MLP model to reverse the spectral degradation
    """

    h, w, C = lr_hsi.shape
    model = SpectralSR_MLP(num_outputs=C, hidden_size=hidden_size)

    # Helper to compute LR at epoch
    def get_lr(epoch):
        if lr_schedule == "one_cycle":
            pct_up = 0.3
            if epoch < pct_up * epochs:
                return init_lr + (max_lr - init_lr) * (epoch / (pct_up * epochs))
            else:
                return max_lr - (max_lr - final_lr) * ((epoch - pct_up * epochs) / ((1 - pct_up) * epochs))
        elif lr_schedule == "cosine_restart":
            period = epochs // num_restarts
            cur = epoch % period
            cos_decay = 0.5 * (1 + math.cos(math.pi * cur / period))
            return min_lr + (max_lr - min_lr) * cos_decay
        else:  # flat
            return init_lr

    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    pbar = tqdm(range(1, epochs+1), desc="Training SR‑MLP", unit="epoch")
    for epoch in pbar:
        lr = get_lr(epoch)
        optimizer.learning_rate.assign(lr)

        # full‑image
        if batch_size is None:
            with tf.GradientTape() as tape:
                pred  = model(lr_msi)
                loss  = loss_fn(lr_hsi, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            pbar.set_postfix(loss=f"{loss.numpy():.4f}", lr=f"{lr:.2e}")

        # tiled
        else:
            # iterate over spatial windows
            epoch_loss = 0.0
            count      = 0
            for y0 in range(0, h, batch_size):
                for x0 in range(0, w, batch_size):
                    sub_msi = lr_msi[y0:y0+batch_size, x0:x0+batch_size, :]
                    sub_hsi = lr_hsi[y0:y0+batch_size, x0:x0+batch_size, :]
                    with tf.GradientTape() as tape:
                        pred  = model(sub_msi)
                        loss  = loss_fn(sub_hsi, pred)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    epoch_loss += loss.numpy()
                    count      += 1

            # report average patch loss
            avg_loss = epoch_loss / count if count else 0.0
            pbar.set_postfix(average_loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

    return model

def run_pipeline(
    HR_MSI,
    LR_HSI,
    srf,
    lr_schedule="one_cycle",
    initial_lr=1e-3,
    max_lr=1e-2,
    final_lr=1e-6,
    min_lr=1e-6,
    num_restarts=2,
    sin_hidden_size=64,
    num_epochs=2500,
    training_batch_size=None,
    inference_batch_size=None,
    synthetic=False
):    
    """
    Runs the entire SpectraLift pipeline end to end.
    
    Parameters:
        HR_MSI (np.ndarray): The high spatial resolution multispectral image (H,W,c)
        LR_HSI (np.ndarray): The low spatial resolution hyperspectral image (h,w,C)
        srf (np.ndarray) of shape (num_bands_msi, num_bands_hsi): The SRF used to degrade the ground truth during the HR MSI generation
    """

    # Start timing
    start_time = time.perf_counter()

    # Obtaining the tensorflow tensors of all the required inputs
    hr_msi, lr_hsi, lr_msi, srf = prepare_inputs(HR_MSI, LR_HSI, srf) #tf.Tensors, hr_msi (H,W,c), lr_hsi (h,w,C), lr_msi (h,w,c)

    # Train the MLP on lr_msi to reverse the spectral degradations
    trained_spectral_sr_mlp = train_spectral_mlp(
        lr_msi, lr_hsi, epochs=num_epochs, lr_schedule=lr_schedule,
        init_lr=initial_lr, max_lr=max_lr, final_lr=final_lr, min_lr=min_lr, num_restarts=num_restarts, hidden_size=sin_hidden_size, batch_size=training_batch_size
    )

    # End timing
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Training completed in {total_time: .2f} seconds")

    if inference_batch_size is None:
        print("Inferring on all input pixels...")
        SR_image, num_params, flops, mem_used, inference_time  = infer_and_analyze_model_performance_tf(
            trained_spectral_sr_mlp,
            sample_inputs=[hr_msi]
        )
        print(f"Parameters:      {num_params:,}")
        print(f"FLOPs:           {flops:,}")
        print(f"GPU Memory:      {mem_used:.2f} MB")
        print(f"Inference time:  {inference_time:.4f} sec")
        SR_image = tf_to_numpy(SR_image)
    else:
        print("Inferring on ", inference_batch_size*inference_batch_size, " pixels per batch...")
        SR_image = batched_inference(
            trained_spectral_sr_mlp, HR_MSI, inference_batch_size, synthetic=synthetic
        )

    # Ouputting the super resolved image (HR HSI)
    SR_image = np.clip(SR_image, 0, 1)

    return SR_image