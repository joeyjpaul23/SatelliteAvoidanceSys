"""End-to-end conjunction-assessment pipeline.

Ingests a catalog, screens for close approaches, assesses collision
probability, and plans fleet maneuvers. The default source is CelesTrak;
synthetic data is a separate, dual-gated path and is never a fallback.
"""

from .cli import main
from .config import PipelineConfig, PipelineResult
from .errors import PipelineError
from .export import plan_artifact, write_plan_json, write_plan_oems, write_plan_text
from .run import load_starlink_slice, run_pipeline

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "PipelineError",
    "run_pipeline",
    "load_starlink_slice",
    "plan_artifact",
    "write_plan_json",
    "write_plan_text",
    "write_plan_oems",
]
