# Quick Start: ROI CSV Generator

## 1️⃣ Installation

```bash
# Ensure you have the required dependencies
pip install pydicom pandas tqdm
```

## 2️⃣ Basic Usage

```bash
python scripts/generate_roi_csv.py --dataset-path /path/to/your/dataset
```

The script will:
- Scan your dataset for DICOM files
- Extract all ROI names
- Display available options
- Ask which options you want to generate
- Create CSV files in `dataset_path/roi_csv/`

## 3️⃣ Dataset Structure

Your dataset must be organized as:
```
your_dataset/
├── PatientID_001/
│   ├── CT/rtstruct.dcm
│   └── MR_T1/rtstruct.dcm
├── PatientID_002/
│   └── CT/rtstruct.dcm
```

## 4️⃣ Four Generation Options

| Option | Description | When to Use |
|--------|-------------|------------|
| **A** | Single ROI per patient | Standard single-ROI analysis |
| **B** | All ROI combinations* | Exploring different ROI selections |
| **C** | All ROIs combined | Comprehensive tumor analysis |
| **D** | All ROI subtractions | Edge/boundary analysis |

*Available if < 10 unique ROIs found

## 5️⃣ Example Output

**Option A:** Each patient gets the same ROI
```csv
PatientID,ImagingScanName,ImagingModality,ROIname
Patient_001,CT,UnknownModality,{GTV_Mass}
Patient_002,CT,UnknownModality,{GTV_Mass}
```

**Option C:** Multiple ROIs combined
```csv
PatientID,ImagingScanName,ImagingModality,ROIname
Patient_001,CT,UnknownModality,{GTV_Mass}+{GTV_Edema}
Patient_002,CT,UnknownModality,{GTV_Mass}
```

**Option D:** ROI subtractions
```csv
PatientID,ImagingScanName,ImagingModality,ROIname
Patient_001,CT,UnknownModality,{GTV_Edema}-{GTV_Mass}
Patient_002,CT,UnknownModality,{GTV_Mass}
```

## 6️⃣ After Generation

1. **Update ImagingModality** (set to actual modality: `CTscan`, `MRscan`, `PTscan`)
2. **Select the best option** for your analysis
3. **Use with MEDiml:**
   ```python
   from MEDiml.wrangling import DataManager
   dm = DataManager(
       config_path='config.yml',
       roi_csv='path/to/roiNames_Tumor.csv'
   )
   ```

## 7️⃣ Command Line Options

```bash
# Interactive mode (recommended)
python scripts/generate_roi_csv.py --dataset-path /data/dataset

# Non-interactive with custom output
python scripts/generate_roi_csv.py \
    --dataset-path /data/dataset \
    --options A C D \
    --roi-label Tumor \
    --output-dir /output/roi_csv
```

## 📚 Full Documentation

See [GENERATE_ROI_CSV_GUIDE.md](GENERATE_ROI_CSV_GUIDE.md) for comprehensive documentation including:
- Detailed option explanations
- Advanced workflows
- Troubleshooting
- API reference
- Examples

## 💡 Tips

✅ Start with **Option A** for your first analysis  
✅ Use **Option C** to include all anatomical structures  
✅ Use **Option D** for edge/boundary analysis  
✅ Always update `ImagingModality` after generation  
✅ Check `generation_summary_*.json` for details  

## ❓ Common Issues

**No ROIs found?**
- Verify DICOM files are RT Structure Sets
- Check file organization matches expected structure
- Test: `python -c "import pydicom; d=pydicom.dcmread('file.dcm'); print(d.StructureSetROISequence)"`

**Too many combinations?**
- Dataset has >10 unique ROI names
- Use Option A or C instead
- Or filter to most relevant ROIs

**ImagingModality is "UnknownModality"?**
- This is expected—update manually based on your scan types
- See documentation for example mapping code
