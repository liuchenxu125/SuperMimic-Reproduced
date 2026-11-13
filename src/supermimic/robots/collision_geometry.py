"""Small geometry fits shared by the Casbot02 collision-asset builder."""
from __future__ import annotations

import numpy as np


def fit_wrist_capsule(points: np.ndarray, padding: float = .001):
    """Enclose the bare wrist shell with a z-axis capsule, not a palm capsule.

    Use a millimetre-rounded radius and solve the shortest axial segment that
    contains every mesh vertex with at least ``padding`` of radial clearance.
    The radius changes with the real wrist shell rather than keeping the old,
    narrower palm approximation.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Expected a finite Nx3 wrist mesh")
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("padding must be positive")
    center = (points.min(0) + points.max(0)) / 2
    radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    radius = float(np.ceil((float(radial.max()) + padding) * 1000) / 1000)
    inner_radius = radius - padding
    reach = np.sqrt(np.maximum(0, inner_radius**2 - radial**2))
    low = float(np.min(points[:, 2] + reach))
    high = float(np.max(points[:, 2] - reach))
    if low > high:
        low = high = (low + high) / 2
    segment = np.array([center[0], center[1], low, center[0], center[1], high])
    return radius, segment
