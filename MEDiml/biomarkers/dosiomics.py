from typing import Optional, Union

import numpy as np

from ..processing.segmentation import compute_bounding_box


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


def coverage(volume: np.ndarray, mask: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], presc_dose: float) -> float:
    """proportion of the tumor volume that is covered by the PIV"""
    tv_piv_val = tv_piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask)  # PIV in cc
    tv_val = tv(volume=volume, vox_dim=vox_dim, mask=mask)  # TV in cc
    return _safe_divide(tv_piv_val, tv_val) * 100.0  # Return percentage


def d_max(volume: np.ndarray, mask: np.ndarray) -> float:
    """Compute Dmax (maximum dose), i.e. 100th percentile dose in ROI (Gy)."""
    return float(np.percentile(_roi_dose_values(volume, mask), 100.0))

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


def selectivity(volume: np.ndarray, mask: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], presc_dose: float) -> float:
    """proportion of the tumor volume that is covered by the PIV"""
    tv_piv_val = tv_piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask)  # PIV in cc
    piv_val = piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask)  # PIV in cc
    return _safe_divide(tv_piv_val, piv_val)


def v_x(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    x: float,
    mask: np.ndarray = None,
    mask_extension: int = 0,
) -> float:
    """
    Compute absolute volume receiving at least x Gy.

    Args:
        volume (np.ndarray): 3D planned dose distribution in Gy.
        vox_dim (list, tuple, np.ndarray): Voxel dimensions in mm as (dx, dy, dz).
        x (float): Dose threshold in Gy.
        mask (np.ndarray, optional): Optional 3D binary mask; if provided, computation
            is restricted to mask voxels > 0.
        mask_extension (int): Extension distance in voxels. If > 0, the bounding box of the mask will be extended
            by this many voxels in each direction

    Returns:
        float: Absolute volume in cc receiving at least x Gy.
    """
    mask_extension = 0 if mask_extension is None else mask_extension
    if mask is not None and mask_extension > 0:
        box_bound = compute_bounding_box(mask=mask)
        extended_mask = np.zeros_like(mask, dtype=bool)

        
        if mask_extension < 1:
            # Percentage based extension
            new_mask = np.array([
                [max(0, box_bound[0][0] - int(box_bound[0][0] * mask_extension)), min(mask.shape[0], box_bound[0][1] + int(box_bound[0][1] * mask_extension))],
                [max(0, box_bound[1][0] - int(box_bound[1][0] * mask_extension)), min(mask.shape[1], box_bound[1][1] + int(box_bound[1][1] * mask_extension))],
                [max(0, box_bound[2][0] - int(box_bound[2][0] * mask_extension)), min(mask.shape[2], box_bound[2][1] + int(box_bound[2][1] * mask_extension))],
            ])

        else:
            # Extend boudning box by the given number of voxels in each direction
            new_mask = np.array([
                [max(0, box_bound[0][0] - mask_extension), min(mask.shape[0], box_bound[0][1] + mask_extension)],
                [max(0, box_bound[1][0] - mask_extension), min(mask.shape[1], box_bound[1][1] + mask_extension)],
                [max(0, box_bound[2][0] - mask_extension), min(mask.shape[2], box_bound[2][1] + mask_extension)],
            ])
        extended_mask[new_mask[0][0]:new_mask[0][1], new_mask[1][0]:new_mask[1][1], new_mask[2][0]:new_mask[2][1]] = 1

        vx_voxel_count = _count_threshold_voxels(volume=volume, threshold=x, mask=extended_mask)
        return float(vx_voxel_count * _voxel_volume_cc(vox_dim))

    vx_voxel_count = _count_threshold_voxels(volume=volume, threshold=x, mask=mask)
    return float(vx_voxel_count * _voxel_volume_cc(vox_dim))


def v_x_p(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    x: float,
    mask: np.ndarray,
    mask_extension: int = 0,
) -> float:
    """
    Compute volume percentage receiving at least x Gy.

    Args:
        volume (np.ndarray): 3D planned dose distribution in Gy.
        vox_dim (list, tuple, np.ndarray): Voxel dimensions in mm as (dx, dy, dz).
        x (float): Dose threshold in Gy.
        mask (np.ndarray): 3D binary mask; computation is restricted to mask voxels > 0.
        mask_extension (int): Extension distance in voxels for the mask bounding box. If > 0, 
            the bounding box of the mask will be extended

    Returns:
        float: Volume percentage receiving at least x Gy.
    """
    vx_val = v_x(volume=volume, vox_dim=vox_dim, x=x, mask=mask, mask_extension=mask_extension)
    tv_val = tv(volume=volume, vox_dim=vox_dim, mask=mask)
    return _safe_divide(vx_val, tv_val) * 100.0  # Return percentage


def tv(volume: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], mask: np.ndarray) -> float:
    """Compute target volume (TV) in cc from ROI mask."""
    _validate_volume(volume)
    _validate_mask(mask, volume)
    tv_voxels = int(np.count_nonzero(mask > 0))
    return float(tv_voxels * _voxel_volume_cc(vox_dim))


def piv(volume: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], presc_dose: float, mask: np.ndarray = None, mask_extension: int = 0) -> float:
    """Compute prescription isodose volume (PIV) in cc over the whole grid."""
    return v_x(volume=volume, vox_dim=vox_dim, x=presc_dose, mask=mask, mask_extension=mask_extension)


def piv_half(volume: np.ndarray, vox_dim: Union[list, tuple, np.ndarray], presc_dose: float, mask_extension: int = 0) -> float:
    """Compute half-prescription isodose volume (PIV_half) in cc over the whole grid."""
    return v_x(volume=volume, vox_dim=vox_dim, x=presc_dose / 2.0, mask=None, mask_extension=mask_extension)


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
    piv_val = piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask)
    tv_piv_val = tv_piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask)
    return _safe_divide(tv_piv_val**2, tv_val * piv_val)


def gradient_index(
    volume: np.ndarray,
    vox_dim: Union[list, tuple, np.ndarray],
    presc_dose: float,
    mask_extension: int = 0
) -> float:
    """
    Compute gradient index.

    GI = PIV_half / PIV
    """
    piv_val = piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask_extension=mask_extension)
    piv_half_val = piv_half(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask_extension=mask_extension)
    return _safe_divide(piv_half_val, piv_val)


def heterogeneity_index_RTOG(volume: np.ndarray, mask: np.ndarray, presc_dose: float) -> float:
    """
    Compute DVH-based heterogeneity index.

    HI = Dmax / Dprescribed
    """
    dmax = d_max(volume=volume, mask=mask)
    return _safe_divide(dmax, presc_dose)


def heterogeneity_index_ICRU_83(volume: np.ndarray, mask: np.ndarray) -> float:
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
    mask: np.ndarray,
    presc_dose: float = None,
    mask_extension: int = 0,
    vx_thresholds: Optional[Union[list, tuple, np.ndarray]] = None,
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

    if presc_dose is not None and presc_dose <= 0:
        raise ValueError("Prescribed dose must be strictly positive")

    if mask_extension is None:
        mask_extension = 0

    if presc_dose is None:
        dosiomics = {
            "Fdos_D2": d2_percent(volume=volume, mask=mask),
            "Fdos_D98": d98_percent(volume=volume, mask=mask),
            "Fdos_D95": d95_percent(volume=volume, mask=mask),
            "Fdos_PIV": None,
            "Fdos_PIV_half": None,
            "Fdos_TV_PIV": None,
            "Fdos_Coverage": None,
            "Fdos_Selectivity": None,
            "Fdos_CI": None,
            "Fdos_GI": None,
            "Fdos_HI_method_ICRU-83": heterogeneity_index_ICRU_83(volume=volume, mask=mask),
            "Fdos_HI_method_RTOG": None,
        }
    else:
        dosiomics = {
            "Fdos_D2": d2_percent(volume=volume, mask=mask),
            "Fdos_D98": d98_percent(volume=volume, mask=mask),
            "Fdos_D95": d95_percent(volume=volume, mask=mask),
            "Fdos_PIV": piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask_extension=mask_extension),
            "Fdos_PIV_half": piv_half(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask_extension=mask_extension),
            "Fdos_TV_PIV": tv_piv(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask),
            "Fdos_Coverage": coverage(volume=volume, mask=mask, vox_dim=vox_dim, presc_dose=presc_dose),
            "Fdos_Selectivity": selectivity(volume=volume, mask=mask, vox_dim=vox_dim, presc_dose=presc_dose),
            "Fdos_CI": conformity_index(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask=mask),
            "Fdos_GI": gradient_index(volume=volume, vox_dim=vox_dim, presc_dose=presc_dose, mask_extension=mask_extension),
            "Fdos_HI_method_ICRU-83": heterogeneity_index_ICRU_83(volume=volume, mask=mask),
            "Fdos_HI_method_RTOG": heterogeneity_index_RTOG(volume=volume, mask=mask, presc_dose=presc_dose),
        }

    if vx_thresholds is None:
        vx_thresholds = [2, 4, 8, 10, 12, 15, 20, 25, 30]

    for x in vx_thresholds:
        dosiomics[f"Fdos_V{x}_cc"] = v_x(volume=volume, vox_dim=vox_dim, x=float(x), mask=mask, mask_extension=mask_extension)
        dosiomics[f"Fdos_V{x}_p"] = v_x_p(volume=volume, vox_dim=vox_dim, x=float(x), mask=mask, mask_extension=mask_extension)

    return dosiomics
