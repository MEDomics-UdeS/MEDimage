#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import dateutil.parser
import numpy as np


class PETSUVConverter:
    """
    A class for converting raw PET volumes into Standardized Uptake Value (SUV) maps.
    """

    def __init__(self, dicom_proxy):
        """
        Initializes the converter with a DICOM proxy object.
        """
        self.dcm = dicom_proxy
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Strategy pattern for dynamic computation routing
        self._strategies = {
            'gml': self._compute_gml,
            'bqml': self._compute_bqml,
            'cm2ml': self._compute_cm2ml,
            'cnts': self._compute_cnts,
            'cps': self._compute_cps
        }

    # ========================================== #
    #                PROPERTIES                  #
    # ========================================== #

    @property
    def unit(self) -> str:
        return str(self.dcm.get(0x00541001, 'unknown')).lower()

    @property
    def patient_weight_g(self) -> float:
        """Returns patient weight_kg in grams."""
        return float(self.dcm[0x0010, 0x1030].value) * 1000.0 if (0x0010, 0x1030) in self.dcm else 75000.0

    @property
    def patient_height_cm(self) -> float:
        """Returns patient height_m in cm."""
        return float(self.dcm[0x0010, 0x1020].value) * 100.0 if (0x0010, 0x1020) in self.dcm else 170.0

    @property
    def patient_sex(self) -> str:
        return str(self.dcm[0x0010, 0x0040].value).upper() if (0x0010, 0x0040) in self.dcm else 'O'

    # ========================================== #
    #              PUBLIC METHODS                #
    # ========================================== #

    def get_conversion_factors(self) -> tuple:
        """
        Calculates conversion factors (suv_factor, rescale_slope, rescale_intercept).
        """
        nuclide_dose = self.dcm[0x0054, 0x0016][0][0x0018, 0x1074].value
        weight_kg = self.patient_weight_g / 1000.0
        half_life = float(self.dcm[0x0054, 0x0016][0][0x0018, 0x1075].value)

        series_time = str(self.dcm[0x0008, 0x0031].value)
        series_date = str(self.dcm[0x0008, 0x0021].value)
        series_dt = dateutil.parser.parse(f"{series_date} {series_time}")

        nuclide_time = str(self.dcm[0x0054, 0x0016][0][0x0018, 0x1072].value)
        nuclide_dt = dateutil.parser.parse(f"{series_date} {nuclide_time}")

        delta_time = (series_dt - nuclide_dt).total_seconds()
        decay_correction = 2 ** (-1 * delta_time / half_life)
        suv_factor = (weight_kg * 1000) / (decay_correction * nuclide_dose)
        
        rescale_slope = self.dcm[0x0028, 0x1053].value
        rescale_intercept = self.dcm[0x0028, 0x1052].value

        manufacturer = str(self.dcm[0x0008, 0x0070].value).lower()
        
        # Philips private tag logic
        if "philips" in manufacturer and "bqml" not in self.unit:
            if 0x70531000 in self.dcm:
                suv_factor = float(self.dcm[0x7053, 0x1000].value)

        return (suv_factor, rescale_slope, rescale_intercept)

    def compute(self, raw_pet: np.ndarray) -> np.ndarray:
        """
        Main entry point. Routes the raw PET array to the correct computation strategy based on unit.
        """
        strategy = self._strategies.get(self.unit)
        if not strategy:
            raise ValueError(f"Unsupported unit '{self.unit}' for SUV computation.")
        
        return strategy(raw_pet)

    # ========================================== #
    #            COMPUTATION STRATEGIES          #
    # ========================================== #

    def _compute_gml(self, raw_pet: np.ndarray) -> np.ndarray:
        weight_kg = self.patient_weight_g / 1000.0
        sex = self.patient_sex
        suv_type = self.dcm.get(0x00541006, 'unknown').lower()

        lbm = None
        bmi = weight_kg / (self.patient_height_cm ** 2) if self.patient_height_cm > 0 else 0

        if suv_type == 'lbm':
            if sex == 'M':
                lbm = 1.10 * weight_kg - 120 * (weight_kg / self.patient_height_cm) ** 2
            elif sex == 'F' or sex == 'O':
                lbm = 1.07 * weight_kg - 148 * (weight_kg / self.patient_height_cm) ** 2

        elif suv_type == 'lbmjames128':
            if sex == 'M':
                lbm = 1.10 * weight_kg - 128 * (weight_kg / self.patient_height_cm) ** 2
            elif sex == 'F' or sex == 'O':
                lbm = 1.07 * weight_kg - 148 * (weight_kg / self.patient_height_cm) ** 2

        elif suv_type == 'lbmjamna':
            if sex == 'M':
                lbm = 9270 * weight_kg / (6680 + 216 * bmi)
            else:
                lbm = 9270 * weight_kg / (8780 + 244 * bmi)

        elif suv_type == 'ibw':
            if sex == 'M':
                lbm = 48 + 1.06 * (self.patient_height_cm - 152)
            else:
                lbm = 45.5 + 0.91 * (self.patient_height_cm - 152)

        elif suv_type == 'bw':
            return raw_pet

        else:
            raise ValueError(f"Unsupported SUV type '{suv_type}'.")
        
        return (raw_pet / lbm) * weight_kg

    def _compute_cm2ml(self, raw_pet: np.ndarray) -> np.ndarray:
        weight_kg = self.patient_weight_g / 1000.0
        bsa = 0.007184 * (weight_kg ** 0.425) * (self.patient_height_cm ** 0.725)
        return (raw_pet / bsa) * weight_kg / 10

    def _compute_cnts(self, raw_pet: np.ndarray) -> np.ndarray:
        if 0x70531000 in self.dcm and float(self.dcm.get(0x70531000)) != 0:
            return raw_pet * float(self.dcm.get(0x70531000))
        
        if 0x70531009 in self.dcm and float(self.dcm.get(0x70531009)) != 0:
            act_scale = float(self.dcm.get(0x70531009))
            return self._compute_bqml(raw_pet * act_scale)
            
        if 0x00181242 in self.dcm and float(self.dcm.get(0x00181242)) != 0:
            frame_duration_sec = float(self.dcm.get(0x00181242)) / 1000.0
            return self._compute_cps(raw_pet / frame_duration_sec)
        
        raise ValueError("No valid scale factor found for 'cnts' unit in DICOM header.")

    def _compute_cps(self, cps_map: np.ndarray) -> np.ndarray:
        corrected_image_tags = self.dcm.get(0x00280051) if 0x00280051 in self.dcm else []
        is_dcal = "DCAL" in corrected_image_tags

        pixel_spacing = self.dcm.get(0x00280030)
        slice_thickness = self.dcm.get(0x00180050)
        
        if not pixel_spacing or not slice_thickness:
            raise KeyError("Voxel dimensions (0028,0030 or 0018,0050) missing.")

        voxel_vol_ml = (float(pixel_spacing[0]) * float(pixel_spacing[1]) * float(slice_thickness)) / 1000.0

        if is_dcal:
            bqml_map = cps_map / voxel_vol_ml
        else:
            cal_factor = self.dcm.get(0x00541322)
            if cal_factor is None:
                raise ValueError("Image is not DCAL and Dose Calibration Factor (0054,1322) is unknown.")
            bqml_map = (cps_map * float(cal_factor)) / voxel_vol_ml

        return self._compute_bqml(bqml_map)

    def _compute_bqml(self, raw_pet: np.ndarray) -> np.ndarray:
        try:
            scantime = None
            if 0x0009100D in self.dcm:
                scantime = self._parse_time(str(self.dcm[0x0009100D].value))
            else:
                scantime = self._parse_time(str(self.dcm[0x0008, 0x0032].value))

            radio_item = self.dcm[0x0054, 0x0016][0]

            if (0x0018, 0x1072) not in radio_item and (0x0018, 0x1078) in radio_item:
                injection_time = self._parse_time(str(radio_item[0x0018, 0x1078].value)[8:])
            elif (0x0018, 0x1072) in radio_item:
                injection_time = self._parse_time(str(radio_item[0x0018, 0x1072].value))
            else:
                raise KeyError("Radiopharmaceutical Start Time tags missing.")

            half_life = float(radio_item[0x0018, 0x1075].value)
            injected_dose = float(radio_item[0x0018, 0x1074].value)
            decay_correction = str(self.dcm.get(0x00541102, '')).upper()

            if decay_correction == 'ADMIN':
                injected_dose_decay = injected_dose
            elif decay_correction == 'START':
                decay = np.exp(-np.log(2) * (scantime - injection_time) / half_life)
                injected_dose_decay = injected_dose * decay
            elif decay_correction == 'NONE':
                _lambda = np.log(2) / half_life
                injected_dose_decay = injected_dose * np.exp(-_lambda * (scantime - injection_time))
                if 0x00181242 not in self.dcm:
                    raise KeyError("Frame Duration (0018,1242) is required for 'NONE' decay correction but is missing.")
                frame_durantion_sec = float(self.dcm.get(0x00181242, 0)) / 1000.0
                factor = (_lambda * frame_durantion_sec) / (1 - np.exp(-_lambda * frame_durantion_sec))
                injected_dose_decay = injected_dose * (1 / factor) * np.exp(-_lambda * (scantime - injection_time))
            else:
                raise ValueError(f"Unrecognized decay correction status: {decay_correction}")

            raw_pet = raw_pet * self.patient_weight_g / injected_dose_decay

            # Convert MBq to Bq if necessary
            if (np.any(raw_pet > 0) and np.nanmean(raw_pet[raw_pet > 0]) > 100):
                raw_pet = raw_pet / 1_000_000.0

        except Exception as e:
            self.logger.warning(f"Error computing BQML ({e}). Using standard 1.75h fallback decay.")
            decay = np.exp(-np.log(2) * (1.75 * 3600) / 6588)
            raw_pet = raw_pet * self.patient_weight_g / (420000000 * decay)

        return raw_pet

    # ========================================== #
    #              STATIC HELPERS                #
    # ========================================== #

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Helper to convert HHMMSS string to total seconds."""
        if '.' in time_str and len(time_str.split('.')[0]) > 6:
            time_str = time_str[8:] # Handle cases where time is prefixed with date
        time_str = str(time_str).zfill(6)
        hh, mm, ss = float(time_str[0:2]), float(time_str[2:4]), float(time_str[4:6])
        return hh * 3600.0 + mm * 60.0 + ss
