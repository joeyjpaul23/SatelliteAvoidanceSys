"""Command-line entry for ``python -m aegis.pipeline``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..envfile import load_env_file
from ..ingest.sources import DataSource, SyntheticNotAuthorizedError
from .config import PipelineConfig
from .errors import PipelineError
from .export import write_plan_json, write_plan_text
from .run import run_pipeline

__all__ = ["main"]

_SYNTHETIC_CLI_REFUSED = (
    "synthetic was refused: --acknowledge-synthetic and "
    "AEGIS_ALLOW_SYNTHETIC=1 are required"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis.pipeline",
        description="Run the AEGIS conjunction-assessment pipeline.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "SPACETRACK, CELESTRAK, or SYNTHETIC; case-insensitive. Default: "
            "CELESTRAK with --tle-path; otherwise SPACETRACK when "
            "SPACETRACK_USER/SPACETRACK_PASS are set, else CELESTRAK."
        ),
    )
    parser.add_argument(
        "--group",
        default="starlink",
        help="CelesTrak GP group, or Space-Track OBJECT_NAME prefix (ignored with --tle-path).",
    )
    parser.add_argument(
        "--tle-path",
        default=None,
        help="Local TLE file. CelesTrak path only; skips HTTP.",
    )
    parser.add_argument(
        "--acknowledge-synthetic",
        action="store_true",
        help="Required for --source SYNTHETIC, together with AEGIS_ALLOW_SYNTHETIC=1.",
    )
    parser.add_argument(
        "--n-planes",
        type=int,
        default=None,
        help="Synthetic constellation planes (ignored on CelesTrak).",
    )
    parser.add_argument(
        "--sats-per-plane",
        type=int,
        default=None,
        help="Synthetic satellites per plane (ignored on CelesTrak).",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Screening window length in seconds (default 5400).",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Keep only the first N objects after ingest.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the plan artifact to this path: .json for JSON, .txt/.md for text.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    if args.duration_s is not None:
        config.duration_s = args.duration_s
    if args.max_objects is not None:
        config.max_objects = args.max_objects
    return config


def _emit_summary(result, output: str | None) -> None:
    if output:
        if Path(output).suffix.lower() in {".txt", ".md"}:
            write_plan_text(result, output)
        else:
            write_plan_json(result, output)
        return
    print(json.dumps(result.plan.summary()))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    load_env_file()
    if args.source is None and args.tle_path is not None:
        source = DataSource.CELESTRAK  # a local TLE file is the CelesTrak path
    elif args.source is None:
        from ..ingest.spacetrack import credentials_from_env

        has_credentials = credentials_from_env() is not None
        source = DataSource.SPACETRACK if has_credentials else DataSource.CELESTRAK
    else:
        source = str(args.source).strip().upper()
    config = _config_from_args(args)

    if source == DataSource.SYNTHETIC:
        if (
            not args.acknowledge_synthetic
            or os.environ.get("AEGIS_ALLOW_SYNTHETIC") != "1"
        ):
            print(_SYNTHETIC_CLI_REFUSED, file=sys.stderr)
            return 1
        from ..ingest.synthetic import SyntheticAuthorization, SyntheticSpec

        spec_kwargs: dict = {}
        if args.n_planes is not None:
            spec_kwargs["n_planes"] = args.n_planes
        if args.sats_per_plane is not None:
            spec_kwargs["sats_per_plane"] = args.sats_per_plane
        try:
            result = run_pipeline(
                source=source,
                authorization=SyntheticAuthorization(acknowledge_synthetic=True),
                synthetic_spec=SyntheticSpec(**spec_kwargs),
                tle_path=args.tle_path,
                config=config,
            )
        except (SyntheticNotAuthorizedError, PipelineError) as error:
            print(str(error), file=sys.stderr)
            return 1
        except Exception as error:  # noqa: BLE001 - CLI must not dump a fake catalog
            print(str(error), file=sys.stderr)
            return 1
        _emit_summary(result, args.output)
        return 0

    if source not in (DataSource.CELESTRAK, DataSource.SPACETRACK):
        print(
            f"unsupported pipeline source: {args.source!r}",
            file=sys.stderr,
        )
        return 2

    try:
        result = run_pipeline(
            source=source,
            group=args.group,
            tle_path=args.tle_path,
            config=config,
        )
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - network/parse failures stay fatal
        print(str(error), file=sys.stderr)
        return 1
    _emit_summary(result, args.output)
    return 0
