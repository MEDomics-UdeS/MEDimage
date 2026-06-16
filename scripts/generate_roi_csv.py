"""
ROI CSV Generator for MEDiml Feature Extraction

This script automates the creation of ROI (Region of Interest) CSV files for MEDiml's
feature extraction pipeline. It reads DICOM files from a dataset organized by PatientID
and ImagingScanName, extracts available ROI names, and generates multiple CSV options
with different ROI combinations.

Author: MEDiml
License: See LICENSE.md
"""

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import pandas as pd
import pydicom
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ROICSVGenerator:
    """Generate ROI CSV files for MEDiml feature extraction."""

    def __init__(self, dataset_path: Union[str, Path], output_dir: Union[str, Path] = None, dicom: bool = True):
        """
        Initialize the ROI CSV Generator.

        Args:
            dataset_path: Path to the dataset organized by PatientID/ImagingScanName
            output_dir: Directory to save output CSV files (default: dataset_path/roi_csv)
            dicom: Whether to scan for DICOM files (True) or NIfTI files (False)
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir) if output_dir else self.dataset_path / "roi_csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dicom = dicom

        self.roi_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # patient_id -> scan_name -> modality -> [roi_names]
        self.all_unique_rois = set()

    def _is_dicom_file(self, file_path: Path) -> bool:
        """Check if a file is a DICOM file."""
        try:
            return pydicom.misc.is_dicom(str(file_path))
        except Exception:
            return False
        
    def _is_nifti_file(self, file_path: Path) -> bool:
        """Check if a file is a NIfTI file."""
        return file_path.name.endswith(('.nii', '.nii.gz'))

    def _extract_rois_from_dicom(self, dicom_path: Path) -> List[str]:
        """
        Extract ROI names from a DICOM-RT file.

        Args:
            dicom_path: Path to DICOM file

        Returns:
            List of ROI names found in the DICOM file
        """
        try:
            dicom = pydicom.dcmread(str(dicom_path), force=True)

            # Check if this is an RT Structure Set
            if not hasattr(dicom, 'StructureSetROISequence'):
                return []

            roi_names = []
            for roi_sequence in dicom.StructureSetROISequence:
                if hasattr(roi_sequence, 'ROIName'):
                    roi_names.append(roi_sequence.ROIName)

            return roi_names
        except Exception as e:
            logger.warning(f"Error reading DICOM file {dicom_path}: {e}")
            return []
    
    def _extract_rois_from_nifti(self, nifti_path: Path) -> List[str]:
        """
        Extract ROI names from a NIfTI file.

        This is a placeholder function. The actual implementation will depend on how ROIs
        are encoded in the NIfTI files (e.g., as labels in a segmentation mask).

        Args:
            nifti_path: Path to NIfTI file
        Returns:
            List of ROI names found in the NIfTI file
        """
        if "(" not in nifti_path.name or ")" not in nifti_path.name:
            raise ValueError(f"NIfTI file name {nifti_path.name} does not contain ROI" \
                "information in expected format (e.g., 'PatientID_ScanName(ROI_NAME).Modality.nii.gz')")

        mask_files = [f for f in nifti_path.parent.rglob(f"{nifti_path.name.split('(')[0]}*.ROI.nii*")]
        return [mask.name[mask.name.find("(") + 1 : mask.name.find(")")] for mask in mask_files]

    def _extract_modality_from_nifti(self, nifti_path: Path) -> List[str]:
        """
        Extract modality information from a NIfTI file.

        This is a placeholder function. The actual implementation will depend on how modalities
        are encoded in the NIfTI files (e.g., as labels in a segmentation mask).

        Args:
            nifti_path: Path to NIfTI file
        Returns:
            str: Modality information extracted from the file name (e.g., 'CTscan', 'MRscan')
        """
        if "." not in nifti_path.name:
            raise ValueError(f"NIfTI file name {nifti_path.name} does not contain modality information "\
                "in expected format (e.g., 'PatientID_ScanName(ROI_NAME).Modality.nii.gz')")
        return nifti_path.name.replace(".nii.gz", "").replace(".nii", "").split(".")[-1]

    def scan_dataset(self) -> None:
        """
        Scan dataset folder structure and extract ROI information.

        Expected structure:
        dataset_path/
            PatientID1/
                ImagingScanName1/
                    DICOM files (RT Structure Set)
                ImagingScanName2/
                    DICOM files
            PatientID2/
                ...
        """
        logger.info(f"Scanning dataset at {self.dataset_path}")

        patient_dirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            scan_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])

            for scan_dir in scan_dirs:
                scan_name = scan_dir.name
                if self.dicom:
                    dicom_files = [f for f in scan_dir.rglob('*') if f.is_file()]

                    for dicom_file in dicom_files:
                        if self._is_dicom_file(dicom_file):
                            roi_names = self._extract_rois_from_dicom(dicom_file)

                            if roi_names:
                                self.roi_data[patient_id][scan_name].extend(roi_names)
                                self.all_unique_rois.update(roi_names)

                else:
                    nifti_files = [f for f in scan_dir.rglob('*.nii*') if f.is_file()]

                    for nifti_file in nifti_files:
                        if self._is_nifti_file(nifti_file):
                            if '.ROI.' in nifti_file.name:
                                continue  # Skip mask files
                            modality = self._extract_modality_from_nifti(nifti_file)
                            roi_name = self._extract_rois_from_nifti(nifti_file)

                            if roi_name:
                                self.roi_data[patient_id][scan_name][modality] = roi_name
                                self.all_unique_rois.update(name for name in roi_name)

        logger.info(f"Found {len(self.roi_data)} patients")
        logger.info(f"Found {len(self.all_unique_rois)} unique ROI names")
        logger.info(f"Unique ROIs: {sorted(self.all_unique_rois)}")

    def _get_patient_scan_info(self) -> List[Tuple[str, str]]:
        """Get list of (patient_id, scan_name) tuples."""
        info = []
        for patient_id in sorted(self.roi_data.keys()):
            for scan_name in sorted(self.roi_data[patient_id].keys()):
                info.append((patient_id, scan_name))
        return info

    def _get_rois_for_patient_scan(self, patient_id: str, scan_name: str) -> Set[str]:
        """Get unique ROIs for a specific patient and scan."""
        modality = list(self.roi_data[patient_id][scan_name].keys())[0]
        rois = self.roi_data[patient_id][scan_name][modality] if modality in self.roi_data[patient_id][scan_name] else []
        return set(rois), modality if rois else set()

    def _check_multiple_rois(self) -> bool:
        """Check if any patient/scan has multiple ROIs."""
        for patient_id in self.roi_data:
            for scan_name in self.roi_data[patient_id]:
                rois, _ = self._get_rois_for_patient_scan(patient_id, scan_name)
                if len(rois) > 1:
                    return True
        return False

    def generate_single_roi_option(self) -> pd.DataFrame:
        """
        Generate Option A: Single ROI per patient.

        Returns:
            DataFrame with columns: PatientID, ImagingScanName, ImagingModality, ROIname
        """
        logger.info("Generating Option A: Single ROI per patient")
        data = []

        for patient_id, scan_name in self._get_patient_scan_info():
            rois, modality = self._get_rois_for_patient_scan(patient_id, scan_name)
            if rois:
                # Use the first (or only) ROI
                roi = sorted(rois)[0]
                data.append({
                    'PatientID': patient_id,
                    'ImagingScanName': scan_name,
                    'ImagingModality': modality if modality else 'UnknownModality',
                    'ROIname': f'{{{roi}}}'
                })

        return pd.DataFrame(data)

    def generate_roi_combinations(self) -> Dict[str, pd.DataFrame]:
        """
        Generate Option B: All possible ROI combinations if < 10 unique ROIs.

        Returns:
            Dictionary with combination name as key and DataFrame as value
        """
        logger.info("Generating Option B: ROI combinations (if < 10 unique ROIs)")
        combinations_dict = {}

        if len(self.all_unique_rois) > 10:
            logger.warning(
                f"Skipping combinations: {len(self.all_unique_rois)} unique ROIs found "
                "(threshold: 10)"
            )
            return combinations_dict

        # Get all possible single ROI selections for each position
        patient_scan_list = self._get_patient_scan_info()

        # For each patient/scan, track possible ROI selections
        roi_options_per_position = []
        for patient_id, scan_name in patient_scan_list:
            rois, modality = self._get_rois_for_patient_scan(patient_id, scan_name)
            roi_options_per_position.append((patient_id, scan_name, sorted(rois), modality))

        # Generate all combinations
        def generate_all_combinations(options_list, index=0, current_combo=None):
            if current_combo is None:
                current_combo = []

            if index == len(options_list):
                return [current_combo[:]]

            results = []
            patient_id, scan_name, rois, modality = options_list[index]

            for roi in rois:
                current_combo.append((patient_id, scan_name, roi, modality))
                results.extend(generate_all_combinations(options_list, index + 1, current_combo))
                current_combo.pop()

            return results

        all_combos = generate_all_combinations(roi_options_per_position)
        logger.info(f"Generated {len(all_combos)} ROI combinations")

        # Convert combinations to DataFrames (limit to reasonable number)
        max_combos = 50  # Prevent explosion of combinations
        for i, combo in enumerate(all_combos[:max_combos]):
            data = []
            for patient_id, scan_name, roi, modality in combo:
                data.append({
                    'PatientID': patient_id,
                    'ImagingScanName': scan_name,
                    'ImagingModality': modality if modality else 'UnknownModality',
                    'ROIname': f'{{{roi}}}'
                })
            combinations_dict[f'combination_{i+1}'] = pd.DataFrame(data)

        if len(all_combos) > max_combos:
            logger.warning(
                f"Generated {len(all_combos)} combinations but only showing {max_combos} "
                "to avoid explosion. Set different ROI filters if needed."
            )

        return combinations_dict

    def generate_combined_roi_option(self) -> pd.DataFrame:
        """
        Generate Option C: Combine all ROIs for multi-ROI patients.

        Returns:
            DataFrame with combined ROIs (using + operator)
        """
        logger.info("Generating Option C: All ROIs combined (A+B+C...)")
        data = []

        for patient_id, scan_name in self._get_patient_scan_info():
            rois, modality = self._get_rois_for_patient_scan(patient_id, scan_name)

            if rois:
                # Combine all ROIs with +
                combined_roi = '+'.join([f'{{{roi}}}' for roi in rois])
                data.append({
                    'PatientID': patient_id,
                    'ImagingScanName': scan_name,
                    'ImagingModality': modality if modality else 'UnknownModality',
                    'ROIname': combined_roi
                })

        return pd.DataFrame(data)

    def generate_subtraction_options(self) -> Dict[str, pd.DataFrame]:
        """
        Generate Option D: All possible subtractions for multi-ROI patients.

        For patients with 2+ ROIs, generate all pairwise subtractions (A-B and B-A).

        Returns:
            Dictionary with subtraction type and DataFrame
        """
        logger.info("Generating Option D: All possible ROI subtractions (A-B, B-A...)")
        subtraction_dict = {}

        for patient_id, scan_name in self._get_patient_scan_info():
            rois, modality = self._get_rois_for_patient_scan(patient_id, scan_name)
            rois = sorted(rois)

            if len(rois) >= 2:
                # Generate all pairwise subtractions
                for roi_a, roi_b in combinations(rois, 2):
                    # Subtraction A - B
                    subtraction_key = f'subtract_{roi_a}_minus_{roi_b}'
                    if subtraction_key not in subtraction_dict:
                        subtraction_dict[subtraction_key] = []

                    subtraction_dict[subtraction_key].append({
                        'PatientID': patient_id,
                        'ImagingScanName': scan_name,
                        'ImagingModality': modality if modality else 'UnknownModality',
                        'ROIname': f'{{{roi_a}}}-{{{roi_b}}}'
                    })

                    # Subtraction B - A
                    subtraction_key_reverse = f'subtract_{roi_b}_minus_{roi_a}'
                    if subtraction_key_reverse not in subtraction_dict:
                        subtraction_dict[subtraction_key_reverse] = []

                    subtraction_dict[subtraction_key_reverse].append({
                        'PatientID': patient_id,
                        'ImagingScanName': scan_name,
                        'ImagingModality': 'UnknownModality',
                        'ROIname': f'{{{roi_b}}}-{{{roi_a}}}'
                    })

        # Convert to DataFrames
        result_dict = {}
        for key, data in subtraction_dict.items():
            result_dict[key] = pd.DataFrame(data)

        logger.info(f"Generated {len(result_dict)} subtraction options")
        return result_dict

    def display_options(self, option_a: pd.DataFrame, option_c: pd.DataFrame,
                       option_b: Dict[str, pd.DataFrame],
                       option_d: Dict[str, pd.DataFrame]) -> None:
        """
        Display all generated options to user for selection.

        Args:
            option_a: Single ROI option DataFrame
            option_c: Combined ROI option DataFrame
            option_b: Dictionary of combination options
            option_d: Dictionary of subtraction options
        """
        print("\n" + "="*80)
        print("ROI CSV GENERATION OPTIONS")
        print("="*80)

        print("\n📋 OPTION A: Single ROI per Patient")
        print("-" * 80)
        print("Each patient/scan gets a single ROI (the first alphabetically).")
        print(f"Number of rows: {len(option_a)}")
        print("\nPreview:")
        print(option_a.head(3).to_string(index=False))

        has_multiple_rois = self._check_multiple_rois()

        if has_multiple_rois:
            print("\n📋 OPTION B: All Possible ROI Combinations")
            print("-" * 80)
            print(f"Number of combination variants: {len(option_b)}")
            if len(self.all_unique_rois) <= 10:
                print(f"All possible combinations generated ({len(option_b)} total)")
            else:
                print(f"Combinations not generated (>{10} unique ROIs)")

            if option_b:
                combo_sample = next(iter(option_b.values()))
                print(f"\nPreview (combination_1, {len(combo_sample)} rows):")
                print(combo_sample.head(3).to_string(index=False))

            print("\n📋 OPTION C: All ROIs Combined")
            print("-" * 80)
            print("For patients with multiple ROIs, all ROIs are combined using (+) operator.")
            print("Example: {GTV_Edema}+{GTV_Mass}")
            print(f"Number of rows: {len(option_c)}")
            print("\nPreview:")
            print(option_c.head(3).to_string(index=False))

            print("\n📋 OPTION D: All Possible Subtractions")
            print("-" * 80)
            print("For patients with multiple ROIs, all pairwise subtractions are generated.")
            print("Example: {GTV_Edema}-{GTV_Mass} and {GTV_Mass}-{GTV_Edema}")
            print(f"Number of subtraction variants: {len(option_d)}")

            if option_d:
                sub_sample = next(iter(option_d.values()))
                subtraction_name = next(iter(option_d.keys()))
                print(f"\nPreview ({subtraction_name}, {len(sub_sample)} rows):")
                print(sub_sample.head(3).to_string(index=False))
        else:
            print("\n⚠️  OPTION B, C, D: Not Available")
            print("-" * 80)
            print("These options require patients with multiple ROIs.")
            print("All patients in this dataset have single ROIs.")

        print("\n" + "="*80)

    def prompt_user_selection(self) -> List[str]:
        """
        Prompt user to select which options to save as CSV files.

        Returns:
            List of selected option identifiers
        """
        selected = []

        print("\n🔧 SELECT OPTIONS TO SAVE")
        print("-" * 80)

        # Always available
        response = input("Save OPTION A (Single ROI per patient)? [y/n]: ").strip().lower()
        if response == 'y':
            selected.append('A')

        has_multiple_rois = self._check_multiple_rois()

        if has_multiple_rois:
            if len(self.all_unique_rois) <= 10:
                response = input("Save OPTION B (All ROI combinations)? [y/n]: ").strip().lower()
                if response == 'y':
                    selected.append('B')

            response = input("Save OPTION C (All ROIs combined)? [y/n]: ").strip().lower()
            if response == 'y':
                selected.append('C')

            response = input("Save OPTION D (All possible subtractions)? [y/n]: ").strip().lower()
            if response == 'y':
                selected.append('D')

        if not selected:
            logger.warning("No options selected!")
            return []

        return selected

    def save_csv_files(self, selected_options: List[str], roi_label: str) -> None:
        """
        Save selected CSV files.

        Args:
            selected_options: List of selected options ('A', 'B', 'C', 'D')
            roi_label: Label for ROI CSV naming (e.g., 'Tumor')
        """
        option_a = self.generate_single_roi_option() if 'A' in selected_options else None
        option_c = self.generate_combined_roi_option() if 'C' in selected_options else None
        option_b = self.generate_roi_combinations() if 'B' in selected_options else {}
        option_d = self.generate_subtraction_options() if 'D' in selected_options else {}

        saved_files = []

        # Save Option A
        if option_a is not None:
            if option_a.empty: raise ValueError("Option A DataFrame is empty. Cannot save CSV.")
            output_file = self.output_dir / f'roiNames_{roi_label}.csv'
            option_a.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")
            saved_files.append(str(output_file))

        # Save Option B combinations
        for i, (combo_name, df) in enumerate(option_b.items(), 1):
            if df.empty: raise ValueError(f"Option B combination '{combo_name}' DataFrame is empty. Cannot save CSV.")
            output_file = self.output_dir / f'roiNames_{roi_label}_combo{i}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")
            saved_files.append(str(output_file))

        # Save Option C
        if option_c is not None:
            if option_c.empty: raise ValueError("Option C DataFrame is empty. Cannot save CSV.")
            output_file = self.output_dir / f'roiNames_{roi_label}_combined.csv'
            option_c.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")
            saved_files.append(str(output_file))

        # Save Option D subtractions
        for subtraction_name, df in option_d.items():
            if df.empty: 
                logging.warning(f"Option D subtraction '{subtraction_name}' DataFrame is empty. Cannot save CSV.")
                continue
            safe_name = subtraction_name.replace(' ', '_')
            output_file = self.output_dir / f'roiNames_{roi_label}_{safe_name}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")
            saved_files.append(str(output_file))

        # Save summary
        summary_file = self.output_dir / f'generation_summary_{roi_label}.json'
        summary = {
            'dataset_path': str(self.dataset_path),
            'roi_label': roi_label,
            'total_patients': len(self.roi_data),
            'unique_rois': sorted(self.all_unique_rois),
            'options_generated': selected_options,
            'files_saved': saved_files
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_file}")

        print(f"\n✅ Successfully saved {len(saved_files)} CSV files to {self.output_dir}")

    def run_interactive(self) -> None:
        """Run the complete interactive ROI CSV generation process."""
        print("\n" + "="*80)
        print("MEDiml ROI CSV Generator")
        print("="*80)

        # Scan dataset
        self.scan_dataset()

        # Generate all options
        option_a = self.generate_single_roi_option()
        option_c = self.generate_combined_roi_option()
        option_b = self.generate_roi_combinations()
        option_d = self.generate_subtraction_options()

        # Display options
        self.display_options(option_a, option_c, option_b, option_d)

        # Get user selection
        selected_options = self.prompt_user_selection()

        if selected_options:
            roi_label = input(
                "\nEnter ROI label for CSV naming (e.g., 'Tumor', 'Brain', 'Lung'): "
            ).strip()
            if not roi_label:
                roi_label = "ROI"

            self.save_csv_files(selected_options, roi_label)
        else:
            logger.warning("No options were selected. Exiting.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate ROI CSV files for MEDiml feature extraction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python generate_roi_csv.py --dataset-path /path/to/dataset

  # With custom output directory
  python generate_roi_csv.py --dataset-path /path/to/dataset --output-dir /path/to/output

  # Non-interactive mode with all options
  python generate_roi_csv.py --dataset-path /path/to/dataset --options A C --roi-label Tumor
        """
    )

    parser.add_argument(
        '--dataset-path',
        required=True,
        type=str,
        help='Path to dataset organized by PatientID/ImagingScanName'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        type=str,
        help='Directory to save output CSV files (default: dataset_path/roi_csv)'
    )
    parser.add_argument(
        '--dicom-or-nifti',
        required=True,
        type=str,
        choices=['dicom', 'nifti'],
        help='Scan for DICOM or NIfTI files (default: scan for DICOM files)'
    )
    parser.add_argument(
        '--options',
        nargs='+',
        choices=['A', 'B', 'C', 'D'],
        default=None,
        help='Options to generate (non-interactive mode). Default: interactive mode'
    )
    parser.add_argument(
        '--roi-label',
        type=str,
        default='ROI',
        help='Label for ROI CSV naming (e.g., "Tumor", "Brain")'
    )

    args = parser.parse_args()

    generator = ROICSVGenerator(args.dataset_path, args.output_dir, args.dicom_or_nifti == 'dicom')

    if args.options:
        # Non-interactive mode
        generator.scan_dataset()
        generator.save_csv_files(args.options, args.roi_label)
    else:
        # Interactive mode
        generator.run_interactive()


if __name__ == '__main__':
    main()
