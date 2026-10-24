"""Explicit paper TF32 configuration, applied before model compilation."""

from __future__ import annotations

import os

import torch


def configure_precision(tf32: bool) -> dict:
    overrides = {
        key: os.environ.get(key)
        for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    }
    if tf32 and overrides["NVIDIA_TF32_OVERRIDE"] == "0":
        raise ValueError("NVIDIA_TF32_OVERRIDE=0 disables requested TF32; unset it first")
    if not tf32 and overrides["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "1":
        raise ValueError("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 contradicts --no-tf32")
    # Use the legacy precision family consistently: the pinned Inductor cache
    # still calls get_float32_matmul_precision(). Mixing in the new per-backend
    # setters makes that call raise on torch 2.14. We do not enable autocast.
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cudnn.allow_tf32 = tf32
    return {
        "tf32_requested": tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "parameter_dtype": "float32",
        "autocast": False,
        "torch_version": torch.__version__,
        "environment_overrides": overrides,
    }
