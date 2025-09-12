This guide explains how to create a Conda environment with all the necessary libraries and dependencies for this repository including those needed for the comparison methods.  
It serves as a fallback in case `requirements.txt` or `environment.yaml` fail to install correctly.

## 1. Create a Conda Environment

Start by creating a new Conda environment with Python 3.9.21:

```bash
conda create --name <env-name> python=3.9.21
conda activate <env-name>
```
Replace <env-name> with your choice of environment name.

## 2. Upgrade Pip and Install TensorFlow with CUDA Support

Once the environment is activated, upgrade `pip` and install TensorFlow (with GPU support via CUDA):

```bash
python -m pip install --upgrade pip
python -m pip install "tensorflow[and-cuda]==2.17.0"
```

## 3. Install PyTorch with CUDA 11.8 Support

Install PyTorch and related libraries with CUDA 11.8 support using the official PyTorch index:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## 4. Install Additional Python Dependencies

Use `pip` to install the remaining libraries required for this project:

```bash
pip install scipy matplotlib tqdm spectral scikit-learn opencv-python==4.11.0.86 fvcore torchsummary torchprofile einops pynvml typing sewar pandas
```

## 5. Install Jupyter Notebook Dependencies (only if you wish to work with jupyter, not required for using the python files)

Use `pip` to install the dependencies in order for jupyter to recognize the conda environment as a kernel

```bash
pip install ipykernel ipywidgets
```

You are now all set and have every dependency required to run any jupyter notebook or python file within this repository.
