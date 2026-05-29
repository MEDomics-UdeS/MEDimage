import argparse
import importlib
from pathlib import Path
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radiomics-nii",
        description="Run radiomics feature extraction on NIfTI files."
    )
    parser.add_argument("path_input", help="Path to the NIfTI input dataset folder.")
    parser.add_argument("path_csv", help="Path to the CSV file with ROI mappings.")
    parser.add_argument("path_settings", help="Path to the radiomics extraction settings file.")
    parser.add_argument("path_save", help="Path where feature files will be written.")
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

    extractor = batch_extractor_cls(
        path_read=Path(args.path_input),
        path_csv=Path(args.path_csv),
        path_params=Path(args.path_settings),
        path_save=Path(args.path_save),
        n_batch=args.n_batch,
        use_niftis=True,
        skip_existing=args.skip_existing,
    )
    extractor.compute_radiomics()


if __name__ == "__main__":
    main()