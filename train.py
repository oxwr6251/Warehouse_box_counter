#!/usr/bin/env python3
"""
Train a YOLO detection model on an Ultralytics-format dataset.

Fine-tunes a pretrained checkpoint (default: yolo11s.pt) for warehouse box
detection. The resulting best.pt weights can be used directly with detect_boxes.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 1280
DEFAULT_BATCH = 4
DEFAULT_DEVICE = "auto"
DEFAULT_PROJECT = "runs/detect"
DEFAULT_NAME = "warehouse_box_detector"


def build_argument_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a YOLO object detection model on an Ultralytics YOLO dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset data.yaml file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Pretrained YOLO checkpoint or architecture (.pt / .yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Input image size for training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Batch size. Use -1 for automatic batch selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Training device: 'auto', 'cpu', 'cuda', or a GPU index like '0'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="Root directory for training run outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Experiment name (subdirectory under --project).",
    )

    return parser


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    print(f"ERROR: {message}", file=sys.stderr, flush=True)


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate CLI argument values.

    Raises:
        ValueError: If any argument is invalid.
    """
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {args.epochs}.")

    if args.imgsz <= 0:
        raise ValueError(f"Image size must be positive, got {args.imgsz}.")

    if args.batch == 0:
        raise ValueError("Batch size cannot be zero. Use a positive integer or -1.")


def load_dataset_config(data_yaml: Path) -> dict[str, Any]:
    """
    Load and parse a YOLO dataset configuration file.

    Args:
        data_yaml: Path to data.yaml.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ValueError: If the file is missing, empty, or invalid YAML.
    """
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    try:
        with data_yaml.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in dataset config '{data_yaml}': {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read dataset config '{data_yaml}': {exc}") from exc

    if not isinstance(config, dict) or not config:
        raise ValueError(f"Dataset config '{data_yaml}' is empty or malformed.")

    if "train" not in config:
        raise ValueError(f"Dataset config '{data_yaml}' must define a 'train' key.")

    if "names" not in config and "nc" not in config:
        raise ValueError(
            f"Dataset config '{data_yaml}' must define 'names' or 'nc' for classes."
        )

    return config


def _resolve_dataset_path(base_dir: Path, value: str) -> Path:
    """Resolve a dataset path relative to the data.yaml directory."""
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _collect_image_paths(source: Path) -> list[Path]:
    """
    Collect image paths from a directory or image list file.

    Args:
        source: Directory path or .txt file listing image paths.

    Returns:
        List of resolved image file paths.
    """
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        )

    if source.is_file() and source.suffix.lower() == ".txt":
        images: list[Path] = []
        with source.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = (source.parent / image_path).resolve()
                images.append(image_path)
        return images

    if source.is_file() and source.suffix.lower() in image_suffixes:
        return [source]

    return []


def validate_dataset(data_yaml: Path) -> dict[str, Any]:
    """
    Validate that the Ultralytics dataset configuration and training data exist.

    Args:
        data_yaml: Path to data.yaml.

    Returns:
        Parsed dataset configuration.

    Raises:
        FileNotFoundError: If required dataset files are missing.
        ValueError: If the dataset configuration is invalid.
    """
    config = load_dataset_config(data_yaml)
    base_dir = data_yaml.parent

    dataset_root = config.get("path", ".")
    if dataset_root is not None:
        root_dir = _resolve_dataset_path(base_dir, str(dataset_root))
    else:
        root_dir = base_dir.resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {root_dir}")

    train_source = _resolve_dataset_path(root_dir, str(config["train"]))
    train_images = _collect_image_paths(train_source)

    if not train_images:
        raise FileNotFoundError(
            f"No training images found for 'train: {config['train']}' "
            f"(resolved to '{train_source}')."
        )

    missing_images = [path for path in train_images if not path.is_file()]
    if missing_images:
        preview = "\n  ".join(str(path) for path in missing_images[:5])
        extra = "" if len(missing_images) <= 5 else f"\n  ... and {len(missing_images) - 5} more"
        raise FileNotFoundError(
            f"{len(missing_images)} training image(s) listed but not found:\n  {preview}{extra}"
        )

    if "val" in config and config["val"]:
        val_source = _resolve_dataset_path(root_dir, str(config["val"]))
        val_images = _collect_image_paths(val_source)
        if not val_images:
            raise FileNotFoundError(
                f"No validation images found for 'val: {config['val']}' "
                f"(resolved to '{val_source}')."
            )

    return config


def ensure_project_directory(project_dir: Path) -> None:
    """
    Create the training project output directory if it does not exist.

    Args:
        project_dir: Root project directory for training runs.

    Raises:
        OSError: If the directory cannot be created.
    """
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create project directory '{project_dir}': {exc}") from exc


def resolve_device(device: str) -> str | int | None:
    """
    Convert CLI device string to an Ultralytics-compatible device value.

    Args:
        device: Device string from CLI ('auto', 'cpu', 'cuda', '0', etc.).

    Returns:
        Device argument for model.train(), or None for automatic selection.
    """
    if device.lower() == "auto":
        return None
    if device.isdigit():
        return int(device)
    return device


def print_training_config(args: argparse.Namespace, data_yaml: Path) -> None:
    """Print selected training settings before starting."""
    separator = "-" * 49
    lines = [
        separator,
        "Training Configuration",
        "",
        "Dataset",
        str(data_yaml.resolve()),
        "",
        "Model",
        args.model,
        "",
        "Epochs",
        str(args.epochs),
        "",
        "Image Size",
        str(args.imgsz),
        "",
        "Batch Size",
        str(args.batch),
        "",
        "Device",
        args.device,
        "",
        "Project",
        args.project,
        "",
        "Run Name",
        args.name,
        separator,
        "",
    ]
    print("\n".join(lines), flush=True)


def print_training_results(save_dir: Path) -> None:
    """
    Print paths to trained model weights and the results directory.

    Args:
        save_dir: Ultralytics training run output directory.
    """
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    separator = "-" * 49

    lines = [
        separator,
        "Training Complete",
        "",
        "Best Model",
        str(best_pt),
        "",
        "Last Model",
        str(last_pt),
        "",
        "Results Directory",
        str(save_dir),
        "",
        "Inference",
        "python detect_boxes.py",
        separator,
    ]
    print("\n".join(lines), flush=True)


def run_training(args: argparse.Namespace) -> int:
    """
    Execute the YOLO training pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 for success, 1 for failure).
    """
    try:
        validate_arguments(args)
    except ValueError as exc:
        print_error(str(exc))
        return 1

    data_yaml = Path(args.data)

    try:
        validate_dataset(data_yaml)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        return 1

    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = Path.cwd() / project_dir

    try:
        ensure_project_directory(project_dir)
    except OSError as exc:
        print_error(str(exc))
        return 1

    print_training_config(args, data_yaml)

    try:
        model = YOLO(args.model)
    except Exception as exc:
        print_error(f"Failed to load model '{args.model}': {exc}")
        return 1

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_dir),
        "name": args.name,
    }

    device = resolve_device(args.device)
    if device is not None:
        train_kwargs["device"] = device

    try:
        results = model.train(**train_kwargs)
    except KeyboardInterrupt:
        print_error("Training interrupted by user.")
        return 1
    except Exception as exc:
        print_error(f"Training failed: {exc}")
        return 1

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project_dir / args.name
    print_training_results(save_dir)
    return 0


def main() -> int:
    """Entry point for the training CLI."""
    parser = build_argument_parser()
    args = parser.parse_args()
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
