"""
Test Dataset Structure Example

This file documents the expected directory structure for datasets to use with generate_roi_csv.py.
It also provides a minimal example that can be used for testing the script.

EXPECTED STRUCTURE:
===================

your_dataset/
├── PatientID_001/
│   ├── CT/
│   │   ├── image_001.dcm
│   │   └── rtstruct.dcm              <- Contains ROI names
│   └── MR_T1/
│       ├── image_001.dcm
│       └── rtstruct.dcm              <- Contains ROI names
│
├── PatientID_002/
│   ├── CT/
│   │   └── rtstruct.dcm
│   └── PET/
│       └── rtstruct.dcm
│
└── PatientID_003/
    └── MR_T2/
        └── rtstruct.dcm


REAL DATASET EXAMPLE (Brain Metastases):
=========================================

brain_mets_dataset/
├── BrainMets-UCSF-00017/
│   └── Dose/
│       ├── dose_plan.dcm
│       └── rtstruct_targets.dcm      <- Contains: {target1}, {target2}, {target3}
│
├── BrainMets-UCSF-00019/
│   └── Dose/
│       └── rtstruct_targets.dcm      <- Contains: {target1}
│
├── BrainMets-UCSF-00035/
│   └── Dose/
│       └── rtstruct_targets.dcm      <- Contains: {target1}, {target2}
│
└── BrainMets-UCSF-00047/
    └── Dose/
        └── rtstruct_targets.dcm      <- Contains: {target1}


REAL DATASET EXAMPLE (Multi-Modality Tumor):
=============================================

tumor_dataset/
├── STS-McGill-001/
│   ├── CT/
│   │   ├── ct_scan_001.dcm
│   │   ├── ct_scan_002.dcm
│   │   └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}, {GTV_Edema}
│   └── MR_T1/
│       ├── mr_t1_001.dcm
│       └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}, {GTV_Edema}
│
├── STS-McGill-002/
│   ├── CT/
│   │   └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}
│   └── PET/
│       └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}
│
├── STS-McGill-003/
│   ├── CT/
│   │   └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}, {GTV_Edema}, {Necrosis}
│   └── MR_T2/
│       └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}, {GTV_Edema}
│
└── STS-McGill-004/
    ├── CT/
    │   └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}
    ├── MR_T1/
    │   └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}, {GTV_Edema}
    └── PET/
        └── rtstruct_tumor.dcm        <- Contains: {GTV_Mass}


UNDERSTANDING ROI EXTRACTION:
=============================

DICOM RT Structure Set Files contain:
- StructureSetROISequence: Array of ROIs
  └─ Each element has:
     - ReferencedROINumber: Unique ID
     - ROIName: Human-readable name (this is what we extract)
     - ReferencedFrameOfReferenceUID: Links to referenced image

Example DICOM access (Python):
    import pydicom
    
    dcm = pydicom.dcmread('rtstruct.dcm')
    
    # List all ROIs
    for roi in dcm.StructureSetROISequence:
        print(f"ROI: {roi.ROIName}")
    
    # Output might be:
    # ROI: GTV_Mass
    # ROI: GTV_Edema
    # ROI: OAR_Spinal_Cord


IMPORTANT NAMING CONVENTIONS:
=============================

PatientID Format:
- Can be: "Patient_001", "STS-McGill-001", "BrainMets-UCSF-00017", "P001", etc.
- Must be consistent within dataset
- Should be meaningful for your study

ImagingScanName Format:
- Modality code: "CT", "PT", "MR_T1", "MR_T2", "Dose", etc.
- Should indicate sequence type
- Used to identify imaging protocol

ROI Name Format (in DICOM):
- Usually: "GTV_Mass", "GTV_Edema", "OAR_Heart", "target1", "PTV", etc.
- Often includes underscores and numbers
- Should match across patients for same structure

Generated CSV Format:
- ROI names wrapped in braces: {GTV_Mass}
- Combinations use + : {GTV_Mass}+{GTV_Edema}
- Subtractions use - : {GTV_Mass}-{GTV_Edema}


EXPECTED GENERATION OUTPUT (from brain_mets_dataset example):
==============================================================

Option A (Single ROI per patient):
PatientID,ImagingScanName,ImagingModality,ROIname
BrainMets-UCSF-00017,Dose,UnknownModality,{target1}
BrainMets-UCSF-00019,Dose,UnknownModality,{target1}
BrainMets-UCSF-00035,Dose,UnknownModality,{target1}
BrainMets-UCSF-00047,Dose,UnknownModality,{target1}

Option C (Combined):
PatientID,ImagingScanName,ImagingModality,ROIname
BrainMets-UCSF-00017,Dose,UnknownModality,{target1}+{target2}+{target3}
BrainMets-UCSF-00019,Dose,UnknownModality,{target1}
BrainMets-UCSF-00035,Dose,UnknownModality,{target1}+{target2}
BrainMets-UCSF-00047,Dose,UnknownModality,{target1}

Option D (Subtractions - examples):
# subtract_target1_minus_target2.csv
PatientID,ImagingScanName,ImagingModality,ROIname
BrainMets-UCSF-00017,Dose,UnknownModality,{target1}-{target2}
BrainMets-UCSF-00019,Dose,UnknownModality,{target1}
BrainMets-UCSF-00035,Dose,UnknownModality,{target1}-{target2}
BrainMets-UCSF-00047,Dose,UnknownModality,{target1}

# subtract_target2_minus_target1.csv
PatientID,ImagingScanName,ImagingModality,ROIname
BrainMets-UCSF-00017,Dose,UnknownModality,{target2}-{target1}
BrainMets-UCSF-00019,Dose,UnknownModality,{target1}
BrainMets-UCSF-00035,Dose,UnknownModality,{target2}-{target1}
BrainMets-UCSF-00047,Dose,UnknownModality,{target1}

... (more subtraction variants)


VERIFICATION CHECKLIST:
=======================

Before running generate_roi_csv.py, verify:

✅ Directory structure follows: PatientID / ScanName / DICOM files
✅ At least one DICOM RT Structure Set file exists per patient/scan
✅ ROI names are consistent across patients (or intentionally different)
✅ DICOM files are readable: 
   ```python
   import pydicom
   dcm = pydicom.dcmread('path/to/file.dcm')
   print(dcm.StructureSetROISequence)
   ```
✅ Sufficient permissions to read all DICOM files
✅ Enough disk space for output CSV files


RUNNING THE SCRIPT:
===================

python scripts/generate_roi_csv.py --dataset-path /path/to/brain_mets_dataset

Expected output:
- Scanning dataset at /path/to/brain_mets_dataset
- Found 4 patients
- Found 3 unique ROI names: {target1, target2, target3}
- [Interactive prompts...]
- Successfully saved 4 CSV files to /path/to/brain_mets_dataset/roi_csv

Output files:
- roiNames_Targets.csv (Option A)
- roiNames_Targets_combined.csv (Option C)
- roiNames_Targets_subtract_target1_minus_target2.csv (Option D)
- roiNames_Targets_subtract_target2_minus_target1.csv (Option D)
- roiNames_Targets_subtract_target1_minus_target3.csv (Option D)
- ... (more combinations)
- generation_summary_Targets.json
"""

if __name__ == '__main__':
    print(__doc__)
