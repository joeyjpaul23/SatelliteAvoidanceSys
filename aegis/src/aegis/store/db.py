"""The experiment ledger: runs, scenarios, results, premiums, artifacts, models.

Backed by stdlib ``sqlite3`` (always available, WAL mode for concurrent
readers/writers) with an optional Postgres path via a lazily imported
``psycopg`` -- the same "no silent fallback" rule as
:mod:`aegis.store.artifacts` applies here: asking for Postgres without
``psycopg`` installed or without a reachable server raises
:class:`~aegis.store.errors.StoreUnavailableError`, it does not quietly
write to sqlite instead.

Why ``record_result`` must be idempotent
-----------------------------------------
A benchmark sweep (``aegis.experiments.run_benchmark``) is resumable: it
skips ``(run_id, scenario_digest, planner)`` triples already recorded so a
crashed or interrupted sweep can restart without re-solving everything. That
only works if re-recording the same triple is safe, so ``results`` carries
a unique index on exactly that triple and every write goes through SQLite's
``INSERT OR REPLACE`` (Postgres: ``INSERT ... ON CONFLICT ... DO UPDATE``),
which deletes-then-inserts on a violation of *any* unique constraint, not
just the primary key. ``result_id`` is a fresh id on every call precisely so
this mechanism -- not manual "does this exist" bookkeeping -- is what
makes the write idempotent.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.timebase import ensure_utc, format_epoch, utc_now
from .artifacts import ArtifactRef
from .errors import StoreUnavailableError

__all__ = ["ExperimentStore"]

# Each entry is (version, statements). Statements are valid SQL in both
# SQLite (3.24+, for the "excluded" upsert alias) and Postgres, so one list
# drives both backends' schema instead of maintaining two dialects.
_MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                git_sha TEXT,
                config_json TEXT NOT NULL,
                code_version TEXT,
                notes TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS scenarios (
                scenario_digest TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                seed INTEGER NOT NULL,
                n_objects INTEGER NOT NULL,
                n_conjunctions INTEGER NOT NULL,
                spec_json TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS results (
                result_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                scenario_digest TEXT NOT NULL,
                planner TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                total_dv_mm_s REAL,
                induced_count INTEGER,
                resolved INTEGER,
                unresolved INTEGER,
                feasible INTEGER,
                solver_time_s REAL,
                created_at TEXT NOT NULL
            )""",
            """CREATE UNIQUE INDEX IF NOT EXISTS results_run_scenario_planner
                ON results (run_id, scenario_digest, planner)""",
            """CREATE TABLE IF NOT EXISTS premiums (
                result_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                scenario_digest TEXT NOT NULL,
                premium REAL,
                dv_fuel_mm_s REAL,
                dv_safe_mm_s REAL,
                infeasible INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT,
                key TEXT NOT NULL,
                uri TEXT NOT NULL,
                digest TEXT NOT NULL,
                content_type TEXT,
                bytes INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS model_versions (
                model_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                architecture_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                artifact_key TEXT,
                train_run_id TEXT
            )""",
        ),
    ),
)


def _empty_distribution(count: int, infeasible_fraction: float) -> dict[str, float | int]:
    return {
        "median": 0.0,
        "p90": 0.0,
        "p99": 0.0,
        "worst": 0.0,
        "mean": 0.0,
        "count": count,
        "infeasible_fraction": infeasible_fraction,
    }


def _premium_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Distribution statistics that are never NaN, even for empty input.

    ``count`` is every recorded premium row (including infeasible ones,
    whose ``premium`` is typically null); the percentile statistics are
    computed only over rows with a defined premium, since an undefined
    premium is not a number to average -- it is a different outcome
    entirely (see ``PremiumRecord`` in the fleetopt.pareto contract).
    """
    count = len(rows)
    if count == 0:
        return _empty_distribution(0, 0.0)

    infeasible_count = sum(1 for row in rows if row.get("infeasible"))
    defined = [float(row["premium"]) for row in rows if row.get("premium") is not None]
    if not defined:
        return _empty_distribution(count, infeasible_count / count)

    values = np.asarray(defined, dtype=float)
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "worst": float(np.max(values)),
        "mean": float(np.mean(values)),
        "count": count,
        "infeasible_fraction": infeasible_count / count,
    }


class ExperimentStore:
    """The run/scenario/result/premium/artifact/model ledger.

    Use as a context manager or call :meth:`close` explicitly::

        with ExperimentStore.open(path) as store:
            store.record_run(run_id="r1", config={})
            ...
    """

    def __init__(self, path: str | Path, *, postgres_dsn: str | None = None) -> None:
        self.path = Path(path)
        self.postgres_dsn = postgres_dsn
        self._conn: sqlite3.Connection | None = None
        self._pg_conn: Any = None
        self._backend = "postgres" if postgres_dsn else "sqlite"

        if postgres_dsn:
            self._pg_conn = self._open_postgres(postgres_dsn)
        else:
            self._conn = self._open_sqlite()

        self._migrate()

    @classmethod
    def open(cls, path: str | Path, *, postgres_dsn: str | None = None) -> ExperimentStore:
        return cls(path, postgres_dsn=postgres_dsn)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._pg_conn is not None:
            self._pg_conn.close()
            self._pg_conn = None

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- connection setup ---------------------------------------------

    def _open_sqlite(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None -> autocommit; every statement is its own
        # transaction, which keeps the retry-on-lock wrapper below simple
        # (there is never a half-finished multi-statement transaction to
        # roll back and retry).
        conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _open_postgres(self, dsn: str) -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise StoreUnavailableError(
                "Postgres experiment store requires the 'psycopg' package, which is not "
                "installed in this environment (pip install 'psycopg[binary]')"
            ) from exc
        try:
            return psycopg.connect(dsn, autocommit=True)
        except Exception as exc:  # noqa: BLE001 - psycopg raises a wide, driver-dependent error family
            raise StoreUnavailableError(f"could not connect to Postgres at the given DSN: {exc}") from exc

    # -- migrations ------------------------------------------------------

    def _migrate(self) -> None:
        self._execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
        )
        rows = self._query("SELECT version FROM schema_version WHERE id = 1")
        current_version = rows[0]["version"] if rows else 0
        for version, statements in _MIGRATIONS:
            if version <= current_version:
                continue
            for statement in statements:
                self._execute(statement)
            self._execute(
                "INSERT INTO schema_version (id, version) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET version = excluded.version",
                (version,),
            )
            current_version = version

    # -- low-level execution ---------------------------------------------

    def _translate(self, sql: str) -> str:
        return sql if self._backend == "sqlite" else sql.replace("?", "%s")

    def _execute(self, sql: str, params: tuple = ()) -> None:
        translated = self._translate(sql)
        if self._backend == "sqlite":
            self._execute_sqlite_with_retry(translated, params)
        else:
            cursor = self._pg_conn.cursor()
            cursor.execute(translated, params)

    def _execute_sqlite_with_retry(
        self, sql: str, params: tuple, *, max_attempts: int = 6, base_delay_s: float = 0.02
    ) -> sqlite3.Cursor:
        assert self._conn is not None
        attempt = 0
        while True:
            try:
                return self._conn.execute(sql, params)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= max_attempts - 1:
                    raise
                time.sleep(base_delay_s * (2**attempt))
                attempt += 1

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        translated = self._translate(sql)
        if self._backend == "sqlite":
            cursor = self._execute_sqlite_with_retry(translated, params)
            return [dict(row) for row in cursor.fetchall()]
        cursor = self._pg_conn.cursor()
        cursor.execute(translated, params)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _upsert(self, table: str, row: dict[str, Any], *, conflict_columns: tuple[str, ...]) -> None:
        columns = tuple(row.keys())
        values = tuple(row[column] for column in columns)
        column_list = ", ".join(columns)
        if self._backend == "sqlite":
            placeholders = ", ".join("?" for _ in columns)
            self._execute(
                f"INSERT OR REPLACE INTO {table} ({column_list}) VALUES ({placeholders})", values
            )
        else:
            placeholders = ", ".join("?" for _ in columns)
            update_columns = [c for c in columns if c not in conflict_columns]
            set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)
            conflict_list = ", ".join(conflict_columns)
            self._execute(
                f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_list}) DO UPDATE SET {set_clause}",
                values,
            )

    @staticmethod
    def _timestamp(created_at: datetime | None) -> str:
        return format_epoch(ensure_utc(created_at) if created_at is not None else utc_now())

    # -- writes ------------------------------------------------------------

    def record_run(
        self,
        *,
        run_id: str,
        git_sha: str | None = None,
        config: dict | None = None,
        code_version: str | None = None,
        notes: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        row = {
            "run_id": run_id,
            "created_at": self._timestamp(created_at),
            "git_sha": git_sha,
            "config_json": json.dumps(config or {}, sort_keys=True),
            "code_version": code_version,
            "notes": notes,
        }
        self._upsert("runs", row, conflict_columns=("run_id",))
        return run_id

    def record_scenario(
        self,
        *,
        scenario_digest: str,
        family: str,
        seed: int,
        n_objects: int,
        n_conjunctions: int,
        spec: dict,
    ) -> str:
        row = {
            "scenario_digest": scenario_digest,
            "family": family,
            "seed": seed,
            "n_objects": n_objects,
            "n_conjunctions": n_conjunctions,
            "spec_json": json.dumps(spec, sort_keys=True),
        }
        self._upsert("scenarios", row, conflict_columns=("scenario_digest",))
        return scenario_digest

    def record_result(
        self,
        *,
        run_id: str,
        scenario_digest: str,
        planner: str,
        metrics: dict,
        total_dv_mm_s: float | None = None,
        induced_count: int | None = None,
        resolved: int | None = None,
        unresolved: int | None = None,
        feasible: bool | None = None,
        solver_time_s: float | None = None,
        result_id: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Record one (run, scenario, planner) result. Idempotent on that triple.

        ``result_id`` is generated fresh unless one is supplied, and the
        write is still idempotent -- see the module docstring for why the
        unique index, not the id, is what SQLite's ``INSERT OR REPLACE``
        keys the replacement on.
        """
        row = {
            "result_id": result_id or uuid.uuid4().hex,
            "run_id": run_id,
            "scenario_digest": scenario_digest,
            "planner": planner,
            "metrics_json": json.dumps(metrics, sort_keys=True),
            "total_dv_mm_s": total_dv_mm_s,
            "induced_count": induced_count,
            "resolved": resolved,
            "unresolved": unresolved,
            "feasible": None if feasible is None else int(feasible),
            "solver_time_s": solver_time_s,
            "created_at": self._timestamp(created_at),
        }
        self._upsert("results", row, conflict_columns=("run_id", "scenario_digest", "planner"))
        return row["result_id"]

    def record_premium(
        self,
        *,
        result_id: str,
        run_id: str,
        scenario_digest: str,
        premium: float | None,
        dv_fuel_mm_s: float | None,
        dv_safe_mm_s: float | None,
        infeasible: bool,
    ) -> str:
        row = {
            "result_id": result_id,
            "run_id": run_id,
            "scenario_digest": scenario_digest,
            "premium": premium,
            "dv_fuel_mm_s": dv_fuel_mm_s,
            "dv_safe_mm_s": dv_safe_mm_s,
            "infeasible": int(infeasible),
        }
        self._upsert("premiums", row, conflict_columns=("result_id",))
        return result_id

    def record_artifact(
        self, *, ref: ArtifactRef, run_id: str | None = None, artifact_id: str | None = None
    ) -> str:
        artifact_id = artifact_id or uuid.uuid4().hex
        row = {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "key": ref.key,
            "uri": ref.uri,
            "digest": ref.digest,
            "content_type": ref.content_type,
            "bytes": ref.bytes_written,
        }
        self._upsert("artifacts", row, conflict_columns=("artifact_id",))
        return artifact_id

    def record_model_version(
        self,
        *,
        architecture: dict,
        metrics: dict,
        model_id: str | None = None,
        artifact_key: str | None = None,
        train_run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        model_id = model_id or uuid.uuid4().hex
        row = {
            "model_id": model_id,
            "created_at": self._timestamp(created_at),
            "architecture_json": json.dumps(architecture, sort_keys=True),
            "metrics_json": json.dumps(metrics, sort_keys=True),
            "artifact_key": artifact_key,
            "train_run_id": train_run_id,
        }
        self._upsert("model_versions", row, conflict_columns=("model_id",))
        return model_id

    # -- queries -----------------------------------------------------------

    def results_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM results WHERE run_id = ? ORDER BY created_at", (run_id,))
        results = []
        for row in rows:
            parsed = dict(row)
            parsed["metrics"] = json.loads(parsed.pop("metrics_json"))
            if parsed["feasible"] is not None:
                parsed["feasible"] = bool(parsed["feasible"])
            results.append(parsed)
        return results

    def premium_distribution(self, run_id: str | None = None) -> dict[str, float | int]:
        if run_id is None:
            rows = self._query("SELECT premium, infeasible FROM premiums")
        else:
            rows = self._query("SELECT premium, infeasible FROM premiums WHERE run_id = ?", (run_id,))
        return _premium_stats(rows)

    def premiums_by_family(self, run_id: str | None = None) -> dict[str, dict[str, float | int]]:
        sql = (
            "SELECT premiums.premium AS premium, premiums.infeasible AS infeasible, "
            "scenarios.family AS family FROM premiums "
            "JOIN scenarios ON premiums.scenario_digest = scenarios.scenario_digest"
        )
        if run_id is None:
            rows = self._query(sql)
        else:
            rows = self._query(f"{sql} WHERE premiums.run_id = ?", (run_id,))

        by_family: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_family.setdefault(row["family"], []).append(row)
        return {family: _premium_stats(group) for family, group in by_family.items()}

    def planner_comparison(self, run_id: str) -> dict[str, dict[str, float | int]]:
        rows = self._query("SELECT * FROM results WHERE run_id = ?", (run_id,))
        by_planner: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_planner.setdefault(row["planner"], []).append(row)

        comparison: dict[str, dict[str, float | int]] = {}
        for planner, group in by_planner.items():
            dvs = [row["total_dv_mm_s"] for row in group if row["total_dv_mm_s"] is not None]
            solver_times = [row["solver_time_s"] for row in group if row["solver_time_s"] is not None]
            feasible_flags = [row["feasible"] for row in group if row["feasible"] is not None]
            comparison[planner] = {
                "count": len(group),
                "mean_total_dv_mm_s": float(np.mean(dvs)) if dvs else 0.0,
                "mean_solver_time_s": float(np.mean(solver_times)) if solver_times else 0.0,
                "feasible_fraction": (sum(bool(f) for f in feasible_flags) / len(feasible_flags))
                if feasible_flags
                else 0.0,
                "resolved_total": sum(row["resolved"] or 0 for row in group),
                "unresolved_total": sum(row["unresolved"] or 0 for row in group),
            }
        return comparison
