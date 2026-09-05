#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import dateutil.parser
import numpy as np

from datetime import datetime, timedelta


def parse_time(time_str):
        """Parse a DICOM time string into a datetime object."""
        if isinstance(time_str, bytes):
            time_str = time_str.decode("utf-8").strip()

        for fmt in ("%H%M%S.%f", "%H%M%S", "%Y%m%d%H%M%S.%f", "%Y%m%d%H%M%S"):
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
            except TypeError:
                continue
        raise ValueError(f"Time data '{time_str}' does not match expected formats")

def calc_elapsed_time(dcm, decay_constant, acquisition_time, injection_time):
    frame_reference_time = float(dcm[0x0054, 0x1300].value) / 1000
    decay_during_frame = decay_constant * dcm.get(0x00181242) / 1000
    avg_count_rate_time = (1 / decay_constant) * np.log(decay_during_frame / (1 - np.exp(-decay_during_frame)))

    return (acquisition_time - injection_time).total_seconds() + avg_count_rate_time - frame_reference_time

def get_injection_time(dcm):
    try:
        return parse_time(dcm[0x0054, 0x0016][0][0x0018, 0x1072].value)
    except AttributeError:
        return parse_time(dcm[0x0054, 0x0016][0][0x0018, 0x1078].value)

def get_tracer_name(rph_item):
    value = getattr(rph_item, "Radiopharmaceutical", None)
    if value is None and (0x0018, 0x0031) in rph_item:
        value = rph_item[(0x0018, 0x0031)].value
    return str(value) if value is not None else None

def get_datetime_on_injection_day(time_value, injection_time):
    return parse_time(time_value).replace(
        year=injection_time.year,
        month=injection_time.month,
        day=injection_time.day,
    )

def compute_elapsed_time_for_start_decay_correction(dcm, injection_time, decay_constant):
    manufacturer = str(dcm[0x0008, 0x0070].value).lower()

    acquisition_time = get_datetime_on_injection_day(str(dcm[0x0008, 0x0032].value), injection_time)
    series_time = get_datetime_on_injection_day(str(dcm[0x0008, 0x0031].value), injection_time)

    if "philips" in manufacturer:
        if acquisition_time == series_time:
            return (acquisition_time - injection_time).total_seconds()
        return calc_elapsed_time(dcm, decay_constant, acquisition_time, injection_time)

    if "siemens" in manufacturer or "cps" in manufacturer or "cti" in manufacturer:
        try:
            private_time = get_datetime_on_injection_day(dcm[(0x0071, 0x1022)].value, injection_time)
            return (private_time - injection_time).total_seconds()
        except (KeyError, TypeError):
            if acquisition_time == series_time:
                return (acquisition_time - injection_time).total_seconds()
            return calc_elapsed_time(dcm, decay_constant, acquisition_time, injection_time)

    if "GE" in manufacturer:
        try:
            private_time = get_datetime_on_injection_day(dcm[(0x0009, 0x100D)].value, injection_time)
            return (private_time - injection_time).total_seconds()
        except (KeyError, TypeError):
            if acquisition_time == series_time:
                return (acquisition_time - injection_time).total_seconds()
            frame_reference_time = float(dcm[0x0054, 0x1300].value) / 1000.0
            return (acquisition_time - injection_time).total_seconds() - frame_reference_time

    if acquisition_time == series_time:
        return (acquisition_time - injection_time).total_seconds()
    return calc_elapsed_time(dcm, decay_constant, acquisition_time, injection_time)


class PETSUVConverter:
    """
    A class for converting raw PET volumes into Standardized Uptake Value (SUV) maps.
    """

    DATE_FORMAT = "%Y%m%d"
    DATE_TIME_FORMAT = "%Y%m%d%H%M%S"

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
        if (0x0010, 0x1030) in self.dcm and float(self.dcm[0x0010, 0x1030].value) > 0:
            weight_kg = float(self.dcm[0x0010, 0x1030].value)
            return weight_kg * 1000.0
        elif float(self.dcm[0x0010, 0x1030].value) == 0:
            raise ValueError("Patient Weight (0010,1030) is zero. Cannot compute SUV.")
        else:
            raise KeyError("Patient Weight (0010,1030) is missing. Cannot compute SUV.")

    @property
    def patient_height_cm(self) -> float:
        """Returns patient height_m in cm."""
        if (0x0010, 0x1020) in self.dcm:
            height_m = float(self.dcm[0x0010, 0x1020].value)
            if height_m <= 0:
                raise ValueError("Patient Height (0010,1020) is zero or negative. Cannot compute SUV.")
            return height_m * 100.0
        else:
            raise KeyError("Patient Height (0010,1020) is missing. Cannot compute SUV.")

    @property
    def patient_sex(self) -> str:
        if (0x0010, 0x0040) not in self.dcm:
            raise KeyError("Patient Sex (0010,0040) is missing. Cannot compute SUV.")
        elif str(self.dcm[0x0010, 0x0040].value).upper() not in ['M', 'F', 'O']:
            raise ValueError(f"Patient Sex (0010,0040) has an invalid value: {self.dcm[0x0010, 0x0040].value}. Expected 'M', 'F', or 'O'. Cannot compute SUV.")
        else:
            return str(self.dcm[0x0010, 0x0040].value).upper()

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

        if suv_type == 'lbm':
            if sex == 'M':
                lbm = 1.10 * weight_kg - 120 * (weight_kg / self.patient_height_cm) ** 2
            elif sex == 'F':
                lbm = 1.07 * weight_kg - 148 * (weight_kg / self.patient_height_cm) ** 2
            else:
                lbm_m = 1.10 * weight_kg - 120 * (weight_kg / self.patient_height_cm) ** 2
                lbm_f = 1.07 * weight_kg - 148 * (weight_kg / self.patient_height_cm) ** 2
                lbm = (lbm_m + lbm_f) / 2

        elif suv_type == 'lbmjames128':
            if sex == 'M':
                lbm = 1.10 * weight_kg - 128 * (weight_kg / self.patient_height_cm) ** 2
            elif sex == 'F':
                lbm = 1.07 * weight_kg - 148 * (weight_kg / self.patient_height_cm) ** 2
            else:
                lbm_m = 1.10 * weight_kg - 128 * (weight_kg / self.patient_height_cm) ** 2
                lbm_f = 1.07 * weight_kg - 148 * (weight_kg / self.patient_height_cm) ** 2
                lbm = (lbm_m + lbm_f) / 2

        elif suv_type == 'lbmjanma':
            bmi = weight_kg / (self.patient_height_cm * 10**-2)**2 if self.patient_height_cm > 0 else 0
            if sex == 'M':
                lbm = (9270 * weight_kg) / (6680 + 216 * bmi)
            elif sex == 'F':
                lbm = (9270 * weight_kg) / (8780 + 244 * bmi)
            else:
                lbm_m = (9270 * weight_kg) / (6680 + 216 * bmi)
                lbm_f = (9270 * weight_kg) / (8780 + 244 * bmi)
                lbm = (lbm_m + lbm_f) / 2

        elif suv_type == 'ibw':
            if sex == 'M':
                lbm = 48 + 1.06 * (self.patient_height_cm - 152)
            elif sex == 'F':
                lbm = 45.5 + 0.91 * (self.patient_height_cm - 152)
            else:
                lbm_m = 48 + 1.06 * (self.patient_height_cm - 152)
                lbm_f = 45.5 + 0.91 * (self.patient_height_cm - 152)
                lbm = (lbm_m + lbm_f) / 2

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

        raise ValueError("No valid scale factor found (SUV scale factor, Activity Concentration Scale Factor or Frame Duration Attribute) for 'cnts' unit in DICOM header.")

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
        # Radiopharmaceutical information
        radio_item = self.dcm[0x0054, 0x0016][0]

        # Safeguard checks for required DICOM tags
        if (0x0018, 0x1075) not in radio_item:
            raise KeyError("Radionuclide Half-Life (0018,1075) is missing.")
        elif float(radio_item[0x0018, 0x1075].value) < 0:
            raise ValueError("Radionuclide Half-Life (0018,1075) is negative. Cannot compute SUV.")

        if (0x0018, 0x1074) not in radio_item:
            raise KeyError("Radionuclide Total Dose (0018,1074) is missing.")
        elif float(radio_item[0x0018, 0x1074].value) < 0:
            raise ValueError("Radionuclide Total Dose (0018,1074) is negative. Cannot compute SUV.")

        # Acquisition Time
        scan_acq_date_time = {
            'date': str(self.dcm[0x0008, 0x0022].value), 
            'time': self._parse_time(self.dcm[0x0008, 0x0032].value)
        }

        # Radiopharmaceutical Start DateTime
        rpsdt = None
        if (0x00181078) in radio_item:
            rpsdt = {
                "date": radio_item[0x0018, 0x1078].value[:8],
                "time": self._parse_time(radio_item[0x0018, 0x1078].value[8:])
            }

        # Radiopharmaceutical Start Time
        if (0x00181072) in radio_item:
            if float(radio_item[0x0018, 0x1072].value) < 0:
                raise ValueError("Radiopharmaceutical Start Time (0018,1072) is negative. Cannot compute SUV.")
            rpst = self._parse_time(str(radio_item[0x0018, 0x1072].value))

        if (0x00181078) not in radio_item and (0x00181072) not in radio_item:
            raise KeyError("Neither Radiopharmaceutical Start DateTime (0018,1078) nor Radiopharmaceutical Start Time (0018,1072) is present. Cannot compute SUV.")

        # Radionuclide Half Life
        half_life = float(radio_item[0x0018, 0x1075].value)
        _lambda = np.log(2) / half_life

        # Administered dose of the radionuclide
        injected_dose = float(radio_item[0x0018, 0x1074].value)

        # Decay correction attribute
        decay_correction = str(self.dcm.get(0x00541102, '')).upper()

        # Series date and time
        series_date_time = {
            'date': str(self.dcm[0x0008, 0x0021].value), 
            'time': self._parse_time(str(self.dcm[0x0008, 0x0031].value))
        }

        # Decay correction logic
        if decay_correction == 'ADMIN':
            injected_dose_decay = injected_dose
            return raw_pet * self.patient_weight_g / injected_dose_decay
        else:
            t_ref = None # decay-correction reference datetime
            t_adm = None # radiopharmaceutical administration datetime
            if decay_correction == 'START':
                manufacturer = str(self.dcm[0x0008, 0x0070].value).lower()
                # If the manufacturer is Siemens or GE, we can use their private tag Decay Correction DateTime (0071,1022)
                if 'siemens' in manufacturer and 0x00711022 in self.dcm:
                    t_ref = {
                        'date': str(self.dcm[0x0071, 0x1022].value)[:8],
                        'time': self._parse_time(str(self.dcm[0x0071, 0x1022].value)[8:])
                    }
                # If the manufacturer is GE, we can use their private tag GEDecayCorrectionDateTime (0009,100D)
                elif 'ge' in manufacturer and 0x0009100D in self.dcm:
                    t_ref = {
                        'date': str(self.dcm[0x0009100D].value)[:8],
                        'time': self._parse_time(str(self.dcm[0x0009100D].value)[8:])
                    }
                elif ('siemens' in manufacturer and 0x00711022 not in self.dcm) or \
                      ('ge' in manufacturer and 0x0009100D not in self.dcm) or \
                        ('ge' not in manufacturer and 'siemens' not in manufacturer):
                    if series_date_time != scan_acq_date_time:
                        if 0x00181242 not in self.dcm:
                            raise KeyError("Frame Duration (0018,1242) is required for 'NONE' decay correction but is missing.")
                        frame_durantion_sec = float(self.dcm.get(0x00181242, 0)) / 1000.0
                        if (0x0054, 0x1300) not in self.dcm:
                            raise KeyError("Frame Reference Time (0054,1300) is missing.")
                        frame_ref_time = float(self.dcm[0x0054, 0x1300].value) / 1000.0
                        if 'ge' in manufacturer:
                            if frame_ref_time < 0:
                                raise ValueError("Frame Reference Time (0054,1300) is negative. Cannot compute SUV.")
                            t_ref = {
                                'date': scan_acq_date_time['date'],
                                'time': scan_acq_date_time['time'] - frame_ref_time
                            }
                        else:
                            if frame_durantion_sec > 0 and frame_ref_time >= 0:
                                # average count rate time in seconds
                                _lambda = np.log(2) / half_life
                                t_ave = (1 / _lambda) * np.log((_lambda*frame_durantion_sec) / (1 - np.exp(-_lambda * frame_durantion_sec)))
                                t_ref = {
                                    'date': scan_acq_date_time['date'],
                                    'time': scan_acq_date_time['time'] - frame_ref_time + t_ave
                                }
                            else:
                                raise ValueError("Frame Duration (0018,1242) or Frame Reference Time (0054,1300) is negative. Cannot compute SUV.")
                    else:
                        t_ref = scan_acq_date_time
            elif decay_correction == 'NONE':
                if 0x00181242 not in self.dcm:
                    raise KeyError("Frame Duration (0018,1242) is required for 'NONE' decay correction but is missing.")
                frame_durantion_sec = float(self.dcm.get(0x00181242, 0)) / 1000.0
                if frame_durantion_sec <= 0:
                    raise ValueError("Frame Duration (0018,1242) is zero or negative. Cannot compute SUV.")
                if half_life <= 0:
                    raise ValueError("Radionuclide Half Life (0018,1075) is zero or negative. Cannot compute SUV.")
                t_ave = (1 / _lambda) * np.log((_lambda*frame_durantion_sec) / (1 - np.exp(-_lambda * frame_durantion_sec)))
                t_ref = {
                    'date': scan_acq_date_time['date'],
                    'time': scan_acq_date_time['time'] + t_ave
                }
            else:
                raise ValueError(f"Unrecognized decay correction status: {decay_correction}")

            if rpsdt:
                # Ensure Radiopharmaceutical Start DateTime (0018,1078) is higher or equal to -3600s 
                # and shorter than twice the Radionuclide Half Life (0018,1075)
                if (-3600 <= self.dt_difference_in_seconds(scan_acq_date_time, rpsdt) < 2 * half_life):
                    t_adm = rpsdt
                else:
                    rpst = rpsdt['time']
                    # arbitrary half-life threshold
                    if half_life < 41400:
                        if (t_ref['time'] - rpst) < -3600:
                            # lower than -3,600s, 24 hours should be subtracted from the resulting datetime
                            t_adm = {
                                "date": self.subtract_days(t_ref['date'], 1), 
                                "time": rpst
                            }
                        else:
                            t_adm = {
                                "date": t_ref['date'], 
                                "time": rpst
                            }
                    else:
                        raise ValueError("Radiopharmaceutical Start DateTime (0018,1078) is unavailable and \
                            the Radionuclide Half Life (0018,1075) is over than 41,400s. Cannot compute SUV.")
            else:
                # arbitrary half-life threshold
                if half_life < 41400:
                    if (t_ref['time'] - rpst) < -3600:
                        # lower than -3,600s, 24 hours should be subtracted from the resulting datetime
                        t_adm = {
                            "date": self.subtract_days(t_ref['date'], 1), 
                            "time": rpst
                        }
                    else:
                        t_adm = {
                            "date": t_ref['date'], 
                            "time": rpst
                        }
                else:
                    raise ValueError("Radiopharmaceutical Start DateTime (0018,1078) is unavailable and \
                        the Radionuclide Half Life (0018,1075) is over than 41,400s. Cannot compute SUV.")

        injected_dose_decay = injected_dose * np.exp(-_lambda * (self.dt_difference_in_seconds(t_ref, t_adm)))
        raw_pet = raw_pet * self.patient_weight_g / injected_dose_decay

        # Convert MBq to Bq if necessary
        if (np.any(raw_pet > 0) and np.nanmean(raw_pet[raw_pet > 0]) >= 10000):
            raw_pet = raw_pet / 1_000_000.0

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

    @staticmethod
    def to_date(date_str):
        """Convert 'YYYYMMDD' string to a datetime object."""
        return datetime.strptime(date_str, PETSUVConverter.DATE_FORMAT)

    @staticmethod
    def to_string(date_obj):
        """Convert a datetime object back to 'YYYYMMDD'."""
        return date_obj.strftime(PETSUVConverter.DATE_FORMAT)

    @staticmethod
    def add_days(date_str, days):
        """Add (or subtract) a number of days."""
        return PETSUVConverter.to_string(PETSUVConverter.to_date(date_str) + timedelta(days=days))

    @staticmethod
    def subtract_days(date_str, days):
        """Subtract a number of days."""
        return PETSUVConverter.add_days(date_str, -days)

    @staticmethod
    def dt_difference_in_seconds(date_time1, date_time2):
        """Calculate the difference in seconds between two datetime objects.
        WARNING: This computes the difference as date_time1 - date_time2, so if date_time1 is earlier than date_time2, the result will be negative.
        """
        def to_datetime(d):
            date = datetime.strptime(d["date"], PETSUVConverter.DATE_FORMAT)
            return date + timedelta(seconds=d["time"])

        dt1 = to_datetime(date_time1)
        dt2 = to_datetime(date_time2)

        return (dt1 - dt2).total_seconds()
