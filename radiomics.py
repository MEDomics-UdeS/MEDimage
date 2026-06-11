import argparse
import importlib
from pathlib import Path
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radiomics",
        description="Run radiomics feature extraction on NIfTI or DICOM files."
    )
    parser.add_argument("path_input", help="Path to the input dataset folder.")
    parser.add_argument("path_csv", help="Path to the CSV file with ROI mappings.")
    parser.add_argument("path_settings", help="Path to the radiomics extraction settings file.")
    parser.add_argument("path_save", help="Path where feature files will be written.")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--use-niftis",
        action="store_true",
        help="Process NIfTI files instead of DICOM files. This is the default."
    )
    format_group.add_argument(
        "--use-dicoms",
        action="store_true",
        help="Process DICOM files instead of NIfTI files."
    )
    parser.add_argument("--n-batch", type=int, default=4, help="Number of CPU cores to use.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scans whose features already exist in the output folder."
    )
    return parser


def _get_batch_extractor():
    module = importlib.import_module("MEDiml.biomarkers.BatchExtractor")
    return module.BatchExtractor


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    batch_extractor_cls = _get_batch_extractor()

    use_niftis = args.use_niftis or not args.use_dicoms

    extractor = batch_extractor_cls(
        path_read=Path(args.path_input),
        path_csv=Path(args.path_csv),
        path_params=Path(args.path_settings),
        path_save=Path(args.path_save),
        n_batch=args.n_batch,
        use_niftis=use_niftis,
        use_dicoms=args.use_dicoms,
        skip_existing=args.skip_existing,
    )
    extractor.compute_radiomics()


if __name__ == "__main__":
    main()