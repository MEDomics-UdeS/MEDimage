from typing import Optional, Union

import numpy as np


def _validate_volume(volume: np.ndarray) -> None:
    """Validate that dose volume is a 3D numpy array."""
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a numpy.ndarray")
    if volume.ndim != 3:
        raise ValueError("volume must be a 3D array")


def _validate_mask(mask: Optional[np.ndarray], volume: np.ndarray) -> None:
    """Validate optional ROI mask against the dose volume geometry."""
    if mask is None:
        return
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray")
    if mask.shape != volume.shape:
        raise ValueError("mask must have the same shape as volume")


def _validate_vox_dim(vox_dim: Union[list, tuple, np.ndarray]) -> np.ndarray:
    """Validate voxel spacing and return it as a float array."""
    vox = np.asarray(vox_dim, dtype=float)
    if vox.shape != (3,):
        raise ValueError("vox_dim must contain exactly 3 elements: [dx, dy, dz] in mm")
    if np.any(vox <= 0):
        raise ValueError("vox_dim elements must be strictly positive")
    return vox


def _voxel_volume_cc(vox_dim: Union[list, tuple, np.ndarray]) -> float:
    """Return voxel volume in cubic centimeters (cc)."""
    vox = _validate_vox_dim(vox_dim)
    return float(np.prod(vox) / 1000.0)  # mm^3 -> cc


def _roi_dose_values(volume: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Extract finite dose values inside ROI mask."""
    _validate_volume(volume)
    _validate_mask(mask, volume)

    if mask is None:
        raise ValueError("mask is required for ROI-based dose metrics")

    roi_vals = volume[mask > 0]
    roi_vals = roi_vals[np.isfinite(roi_vals)]

    if roi_vals.size == 0:
        raise ValueError("ROI is empty or contains no finite dose values")

    return roi_vals


def _count_threshold_voxels(
    volume: np.ndarray,
    threshold: float,
    mask: Optional[np.ndarray] = None,
) -> int:
    """Count finite voxels receiving at least a given threshold."""
    _validate_volume(volume)
    _validate_mask(mask, volume)

    finite = np.isfinite(volume)
    if mask is None:
        condition = finite & (volume >= threshold)
    else:
        condition = finite & (mask > 0) & (volume >= threshold)

    return int(np.count_nonzero(condition))


def _safe_divide(num: float, den: float) -> float:
    """Safely divide two floats; return nan when denominator is zero."""
    if den == 0:
        return float("nan")
    return float(num / den)


def d2_percent(volume: np.ndarray, mask: np.ndarray) -> float:
    """Compute D2% (near-maximum dose), i.e. 98th percentile dose in ROI (Gy)."""
    return float(np.percentile(_roi_dose_values(volume, mask), 98.0))


def d98_percent(volume: np.ndarray, mask: np.ndarray) -> float:
    """Compute D98% (near-minimum dose), i.e. 2nd percentile dose in ROI (Gy)."""
    return float(np.percentile(_roi_dose_values(volume, mask), 2.0))


def d95_percent(volume: np.ndarray, mask: np.ndarray) -> float:
    """Compute D95% (target coverage dose), i.e. 5th percentile dose in ROI (Gy)."""
    return float(np.percentile(_roi_dose_values(volume, mask), 5.0))


def d50_percent(volume: np.ndarray, mask: np.ndarray) -> float:
    """Compute D50% (median dose), i.e. 50th percentile dose in ROI (Gy)."""
    return float(np.percentile(_roi_dose_values(volume, mask), 50.0))


def v_x(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    x: float,
    mask: np.ndarray = None,
) -> float:
    """
    Compute absolute volume receiving at least x Gy.

    Args:
        volume (np.ndarray): 3D planned dose distribution in Gy.
        vox_dim (list, tuple, np.ndarray): Voxel dimensions in mm as (dx, dy, dz).
        x (float): Dose threshold in Gy.
        mask (np.ndarray, optional): Optional 3D binary mask; if provided, computation
            is restricted to mask voxels > 0.

    Returns:
        float: Absolute volume in cc receiving at least x Gy.
    """
    vx_voxel_count = _count_threshold_voxels(volume=volume, threshold=x, mask=mask)
    return float(vx_voxel_count * _voxel_volume_cc(vox_dim))


def tv(volume: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], mask: np.ndarray) -> float:
    """Compute target volume (TV) in cc from ROI mask."""
    _validate_volume(volume)
    _validate_mask(mask, volume)
    tv_voxels = int(np.count_nonzero(mask > 0))
    return float(tv_voxels * _voxel_volume_cc(vox_dim))


def piv(volume: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], presc_dose: float) -> float:
    """Compute prescription isodose volume (PIV) in cc over the whole grid."""
    return v_x(volume=volume, vox_dim=vox_dim, x=presc_dose, mask=None)


def piv_half(volume: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], presc_dose: float) -> float:
    """Compute half-prescription isodose volume (PIV_half) in cc over the whole grid."""
    return v_x(volume=volume, vox_dim=vox_dim, x=presc_dose / 2.0, mask=None)


def tv_piv(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    presc_dose: float,
    mask: np.ndarray,
) -> float:
    """Compute TV_PIV overlap volume (TV intersect PIV) in cc."""
    return v_x(volume=volume, vox_dim=vox_dim, x=presc_dose, mask=mask)


def conformity_index(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    presc_dose: float,
    mask: np.ndarray,
) -> float:
    """
    Compute Paddick conformity index.

    CI = (TV_PIV^2) / (TV * PIV)
    """
    tv_val = tv(volume=volume, vox_dim=vox_dim, mask=mask)
    piv_val = piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose)
    tv_piv_val = tv_piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask)
    return _safe_divide(tv_piv_val**2, tv_val * piv_val)


def gradient_index(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    presc_dose: float,
) -> float:
    """
    Compute gradient index.

    GI = PIV_half / PIV
    """
    piv_val = piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose)
    piv_half_val = piv_half(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose)
    return _safe_divide(piv_half_val, piv_val)


def heterogeneity_index(volume: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute DVH-based heterogeneity index.

    HI = (D2% - D98%) / D50%
    """
    d2 = d2_percent(volume=volume, mask=mask)
    d98 = d98_percent(volume=volume, mask=mask)
    d50 = d50_percent(volume=volume, mask=mask)
    return _safe_divide(d2 - d98, d50)


def extract_all(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    presc_dose: float,
    mask: np.ndarray,
) -> dict:
    """Compute all implemented dosiomics features.

    Definitions:
        - D2%: 98th percentile dose in ROI (Gy)
        - D98%: 2nd percentile dose in ROI (Gy)
        - D95%: 5th percentile dose in ROI (Gy)
        - PIV: absolute volume receiving prescription dose in whole grid (cc)
        - PIV_half: absolute volume receiving half prescription dose in whole grid (cc)
        - TV_PIV: overlap volume TV intersect PIV (cc)
        - CI: Paddick conformity index = (TV_PIV^2)/(TV*PIV)
        - GI: gradient index = PIV_half/PIV
        - HI: heterogeneity index = (D2%-D98%)/D50%
    """
    _validate_volume(volume)
    _validate_mask(mask, volume)
    _validate_vox_dim(vox_dim)

    if presc_dose <= 0:
        raise ValueError("Prescribed dose must be strictly positive")

    dosiomics = {
        "Fdos_D2": d2_percent(volume=volume, mask=mask),
        "Fdos_D98": d98_percent(volume=volume, mask=mask),
        "Fdos_D95": d95_percent(volume=volume, mask=mask),
        "Fdos_PIV": piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose),
        "Fdos_PIV_half": piv_half(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose),
        "Fdos_TV_PIV": tv_piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask),
        "Fdos_CI": conformity_index(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask),
        "Fdos_GI": gradient_index(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose),
        "Fdos_HI": heterogeneity_index(volume=volume, mask=mask),
    }

    return dosiomics
