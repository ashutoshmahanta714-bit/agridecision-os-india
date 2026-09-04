"""Command-line entry point for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agridecision.config import get_data_gov_api_key, load_config
from agridecision.data.csv_loader import load_mandi_file
from agridecision.data.data_gov import DataGovClient
from agridecision.data.demo import generate_demo_mandi_data
from agridecision.data.provenance import create_snapshot_manifest, write_snapshot_manifest
from agridecision.data.quality import validate_mandi_data
from agridecision.data.readiness import assess_training_readiness
from agridecision.data.schema import standardise_mandi_frame
from agridecision.features.tabular import build_supervised_frame
from agridecision.logging_utils import configure_logging
from agridecision.models.training import train_model_suite


def _write_quality_report(report: object, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")  # type: ignore[attr-defined]


def _prepare_frame(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    standardised = standardise_mandi_frame(frame)
    accepted, quarantine, report = validate_mandi_data(standardised)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output_dir / "clean_mandi_prices.csv", index=False)
    quarantine.to_csv(output_dir / "quarantined_rows.csv", index=False)
    _write_quality_report(report, output_dir / "data_quality_report.json")
    coverage, readiness = assess_training_readiness(accepted)
    coverage.to_csv(output_dir / "market_coverage.csv", index=False)
    (output_dir / "training_readiness.json").write_text(
        json.dumps(readiness.to_dict(), indent=2), encoding="utf-8"
    )
    if not readiness.ready_for_forecasting:
        raise ValueError(
            "The data passed row validation but lacks sufficient time history for forecasting: "
            + ", ".join(readiness.reasons)
        )
    supervised = build_supervised_frame(
        accepted,
        horizon_days=args.horizon,
        shock_threshold=args.shock_threshold,
    )
    supervised.to_csv(output_dir / "supervised_features.csv", index=False)
    return supervised


def run_demo(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    raw = generate_demo_mandi_data(days=args.days, seed=args.seed)
    raw.to_csv(output / "synthetic_demo_raw.csv", index=False) if output.exists() else None
    supervised = _prepare_frame(raw, output, args)
    result = train_model_suite(
        supervised,
        output,
        horizon_days=args.horizon,
        random_state=args.seed,
    )
    # Save raw after _prepare_frame has created the directory.
    raw.to_csv(output / "synthetic_demo_raw.csv", index=False)
    write_snapshot_manifest(
        create_snapshot_manifest(
            output / "synthetic_demo_raw.csv",
            source_url="internal://agridecision/data/demo.py",
            source_name="AgriDecision deterministic synthetic fixture",
            license_name="MIT",
            is_synthetic=True,
        ),
        output / "snapshot_manifest.json",
    )
    print(json.dumps(result["metrics"], indent=2, default=str))
    print(f"\nDemo artifacts written to: {output.resolve()}")


def run_prepare(args: argparse.Namespace) -> None:
    frame = load_mandi_file(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_snapshot_manifest(
        create_snapshot_manifest(
            args.input,
            source_url=args.source_url,
            source_name=args.source_name,
            is_synthetic=False,
        ),
        output / "snapshot_manifest.json",
    )
    supervised = _prepare_frame(frame, output, args)
    print(f"Prepared {len(supervised):,} supervised rows")


def run_train(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input, parse_dates=["arrival_date"])
    result = train_model_suite(
        frame,
        args.output_dir,
        horizon_days=args.horizon,
        random_state=args.seed,
    )
    print(json.dumps(result["metrics"], indent=2, default=str))


def run_download(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_config = config["data"]
    client = DataGovClient(
        get_data_gov_api_key(),
        data_config["resource_id"],
        base_url=data_config["api_base_url"],
        page_size=args.page_size,
        request_delay_seconds=args.delay,
    )

    def progress(downloaded: int, total: int | None) -> None:
        suffix = f" / {total:,}" if total is not None else ""
        print(f"Downloaded {downloaded:,}{suffix} records", flush=True)

    filters = {"commodity": args.commodity} if args.commodity else None
    frame = client.download_csv(
        args.output,
        filters=filters,
        max_pages=args.max_pages,
        resume=not args.no_resume,
        progress=progress,
    )
    write_snapshot_manifest(
        create_snapshot_manifest(
            args.output,
            source_url=f"{data_config['api_base_url'].rstrip('/')}/{data_config['resource_id']}",
            source_name="Data.gov.in mandi price resource",
            is_synthetic=False,
        ),
        Path(args.output).with_suffix(".manifest.json"),
    )
    print(f"Saved {len(frame):,} records to {Path(args.output).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agridecision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--output-dir", default="artifacts")
    shared.add_argument("--horizon", type=int, default=7)
    shared.add_argument("--shock-threshold", type=float, default=0.15)
    shared.add_argument("--seed", type=int, default=42)

    demo = subparsers.add_parser("demo", parents=[shared], help="run the synthetic demo")
    demo.add_argument("--days", type=int, default=480)
    demo.set_defaults(func=run_demo)

    prepare = subparsers.add_parser("prepare", parents=[shared], help="validate a CSV/ZIP")
    prepare.add_argument("--input", required=True)
    prepare.add_argument(
        "--source-url",
        default="https://www.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
    )
    prepare.add_argument("--source-name", default="Official Data.gov.in mandi price download")
    prepare.set_defaults(func=run_prepare)

    train = subparsers.add_parser("train", parents=[shared], help="train from prepared features")
    train.add_argument("--input", required=True)
    train.set_defaults(func=run_train)

    download = subparsers.add_parser("download", help="download a Data.gov.in resource")
    download.add_argument("--config", default="configs/base.yaml")
    download.add_argument("--output", default="data/raw/mandi_prices.csv")
    download.add_argument("--commodity", default="Onion")
    download.add_argument("--page-size", type=int, default=500)
    download.add_argument("--delay", type=float, default=1.5)
    download.add_argument("--max-pages", type=int)
    download.add_argument("--no-resume", action="store_true")
    download.set_defaults(func=run_download)
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
