"""Orchestrate the full benchmark run."""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

from benchmark import config
from benchmark import checks, inputs, memory, report, runner, scenarios, timing
from benchmark.environment import probe, write_environment


def generate_run_id() -> str:
    """Generate a unique run ID.

    Returns:
        A string like '2026-10-23T14-30-00-abc123'.
    """
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


def _run_s1_suite(suite_dir: Path, warmups: int, repeats: int) -> list[dict[str, Any]]:
    """Run the S1 suite with 4 cells (B=1,5,10,20).

    Args:
        suite_dir: Suite artifact directory.
        warmups: Number of warmup runs.
        repeats: Number of timed runs.

    Returns:
        List of run records.
    """
    records: list[dict[str, Any]] = []
    raw_log = suite_dir / "raw_runs.jsonl"

    for cell in scenarios.b1_cells():
        batch_size = cell["batch_size"]
        cell_name = cell["cell"]
        cfg = config.s1_config(batch_size, suite_dir.parent)
        cfg_path = suite_dir / "configs" / f"{cell_name}.yaml"
        config.write_yaml(cfg_path, cfg)

        # Warmups (discarded)
        for rep in range(warmups):
            art = suite_dir / "artifacts" / cell_name / "ours" / f"warmup_{rep}"
            tel = art / "telemetry.json"
            record = runner.run_cell(cell_name, "ours", rep, cfg_path, art, tel)
            record["warmup"] = True
            runner.append_raw_log(raw_log, record)
            records.append(record)

        # Timed runs
        for rep in range(repeats):
            # Side A (ours)
            art_a = suite_dir / "artifacts" / cell_name / "ours" / f"run_{rep}"
            tel_a = art_a / "telemetry.json"
            record_a = runner.run_cell(cell_name, "ours", rep, cfg_path, art_a, tel_a)
            runner.append_raw_log(raw_log, record_a)
            records.append(record_a)
            if not checks.check_row_count(art_a, 1000):
                raise RuntimeError(f"Row count mismatch: {cell_name} ours run_{rep}")

            # Side D (direct)
            art_d = suite_dir / "artifacts" / cell_name / "direct" / f"run_{rep}"
            tel_d = art_d / "telemetry.json"
            record_d = runner.run_cell(cell_name, "direct", rep, cfg_path, art_d, tel_d)
            runner.append_raw_log(raw_log, record_d)
            records.append(record_d)
            if not checks.check_row_count(art_d, 1000):
                raise RuntimeError(f"Row count mismatch: {cell_name} direct run_{rep}")

    return records


def _run_s2_suite(suite_dir: Path, warmups: int, repeats: int) -> list[dict[str, Any]]:
    """Run the S2 suite (40 transform types).

    Args:
        suite_dir: Suite artifact directory.
        warmups: Number of warmup runs.
        repeats: Number of timed runs.

    Returns:
        List of run records.
    """
    records: list[dict[str, Any]] = []
    raw_log = suite_dir / "raw_runs.jsonl"
    cfg = config.s2_config(suite_dir.parent)
    cfg_path = suite_dir / "configs" / "s2.yaml"
    config.write_yaml(cfg_path, cfg)

    # Warmups
    for rep in range(warmups):
        art = suite_dir / "artifacts" / "s2" / "ours" / f"warmup_{rep}"
        tel = art / "telemetry.json"
        record = runner.run_cell("s2", "ours", rep, cfg_path, art, tel)
        record["warmup"] = True
        runner.append_raw_log(raw_log, record)
        records.append(record)

    # Timed runs
    for rep in range(repeats):
        art_a = suite_dir / "artifacts" / "s2" / "ours" / f"run_{rep}"
        tel_a = art_a / "telemetry.json"
        record_a = runner.run_cell("s2", "ours", rep, cfg_path, art_a, tel_a)
        runner.append_raw_log(raw_log, record_a)
        records.append(record_a)
        if not checks.check_row_count(art_a, 40):
            raise RuntimeError(f"Row count mismatch: s2 ours run_{rep}")

        art_d = suite_dir / "artifacts" / "s2" / "direct" / f"run_{rep}"
        tel_d = art_d / "telemetry.json"
        record_d = runner.run_cell("s2", "direct", rep, cfg_path, art_d, tel_d)
        runner.append_raw_log(raw_log, record_d)
        records.append(record_d)
        if not checks.check_row_count(art_d, 40):
            raise RuntimeError(f"Row count mismatch: s2 direct run_{rep}")

    return records


def _run_s3_suite(suite_dir: Path, warmups: int, repeats: int) -> list[dict[str, Any]]:
    """Run the S3 suite (40 sigma levels).

    Args:
        suite_dir: Suite artifact directory.
        warmups: Number of warmup runs.
        repeats: Number of timed runs.

    Returns:
        List of run records.
    """
    records: list[dict[str, Any]] = []
    raw_log = suite_dir / "raw_runs.jsonl"
    cfg = config.s3_config(suite_dir.parent)
    cfg_path = suite_dir / "configs" / "s3.yaml"
    config.write_yaml(cfg_path, cfg)

    # Warmups
    for rep in range(warmups):
        art = suite_dir / "artifacts" / "s3" / "ours" / f"warmup_{rep}"
        tel = art / "telemetry.json"
        record = runner.run_cell("s3", "ours", rep, cfg_path, art, tel)
        record["warmup"] = True
        runner.append_raw_log(raw_log, record)
        records.append(record)

    # Timed runs
    for rep in range(repeats):
        art_a = suite_dir / "artifacts" / "s3" / "ours" / f"run_{rep}"
        tel_a = art_a / "telemetry.json"
        record_a = runner.run_cell("s3", "ours", rep, cfg_path, art_a, tel_a)
        runner.append_raw_log(raw_log, record_a)
        records.append(record_a)
        if not checks.check_row_count(art_a, 40):
            raise RuntimeError(f"Row count mismatch: s3 ours run_{rep}")

        art_d = suite_dir / "artifacts" / "s3" / "direct" / f"run_{rep}"
        tel_d = art_d / "telemetry.json"
        record_d = runner.run_cell("s3", "direct", rep, cfg_path, art_d, tel_d)
        runner.append_raw_log(raw_log, record_d)
        records.append(record_d)
        if not checks.check_row_count(art_d, 40):
            raise RuntimeError(f"Row count mismatch: s3 direct run_{rep}")

    return records


def _write_timing_csvs(
    run_dir: Path,
    suites: list[str],
    all_records: list[dict[str, Any]],
) -> None:
    """Write timing CSVs for each suite.

    Args:
        run_dir: Run results directory.
        suites: List of suite names.
        all_records: All run records.
    """
    timing_summary = timing.aggregate_timing(all_records)
    timing_fields = ["cell", "side", "n_timed", "median_s", "iqr_s", "min_s", "max_s"]

    if "s1" in suites:
        b1_rows = [r for r in timing_summary if r["cell"].startswith("b")]
        timing.write_csv(run_dir / "b1_timing.csv", b1_rows, timing_fields)
    if "s2" in suites:
        s2_rows = [r for r in timing_summary if r["cell"] == "s2"]
        timing.write_csv(run_dir / "s2_timing.csv", s2_rows, timing_fields)
    if "s3" in suites:
        s3_rows = [r for r in timing_summary if r["cell"] == "s3"]
        timing.write_csv(run_dir / "s3_timing.csv", s3_rows, timing_fields)


def _write_memory_csvs(
    run_dir: Path,
    suites: list[str],
    all_records: list[dict[str, Any]],
) -> None:
    """Write memory CSVs for each suite.

    Args:
        run_dir: Run results directory.
        suites: List of suite names.
        all_records: All run records.
    """
    memory_summary = memory.aggregate_memory(all_records)
    mem_fields = ["cell", "side", "n_timed", "peak_rss_mib_median", "peak_rss_mib_iqr"]

    if "s1" in suites:
        b1_mem = [r for r in memory_summary if r["cell"].startswith("b")]
        timing.write_csv(run_dir / "b1_memory.csv", b1_mem, mem_fields)
    if "s2" in suites:
        s2_mem = [r for r in memory_summary if r["cell"] == "s2"]
        timing.write_csv(run_dir / "s2_memory.csv", s2_mem, mem_fields)
    if "s3" in suites:
        s3_mem = [r for r in memory_summary if r["cell"] == "s3"]
        timing.write_csv(run_dir / "s3_memory.csv", s3_mem, mem_fields)


def _setup_run_dir(
    outputs_root: Path,
    quick: bool,
    only: str | None,
    with_attribution: bool,
) -> tuple[Path, Path, list[str], int, int]:
    """Set up the run directory structure and determine suites and parameters.

    Args:
        outputs_root: Root directory for outputs.
        quick: If True, use fewer repeats/warmups.
        only: If set, run only this suite.
        with_attribution: If True, run attribution.

    Returns:
        Tuple of (run_dir, artifacts_dir, suites, warmups, repeats).
    """
    run_id = generate_run_id()
    run_dir = outputs_root / "results" / run_id
    artifacts_dir = outputs_root / "artifacts" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Write environment receipt
    env = probe()
    write_environment(run_dir / "environment.json", env)

    # Generate inputs (or use cache)
    inputs_dir = artifacts_dir / "inputs"
    if not (inputs_dir / "manifest.parquet").exists():
        inputs.generate_b1(inputs_dir, count=1000, seed=42)

    # Prepare comma10k input for S2/S3 suites (or use cache)
    if not (inputs_dir / "comma10k.parquet").exists():
        # outputs_root is benchmark/outputs, so .parent.parent.parent = repo root
        samples_path = outputs_root.parent.parent.parent / "examples" / "data" / "comma10k_sample" / "samples.parquet"
        if samples_path.exists():
            inputs.prepare_comma10k(inputs_dir, samples_path)

    # Determine suites to run
    suites = []
    if only is None or only == "s1":
        suites.append("s1")
    if only is None or only == "s2":
        suites.append("s2")
    if only is None or only == "s3":
        suites.append("s3")

    # Determine repeats and warmups
    warmups = 1 if quick else 3
    repeats = 3 if quick else 7

    return run_dir, artifacts_dir, suites, warmups, repeats


def _write_s3_files(run_dir: Path, artifacts_dir: Path) -> None:
    """Generate s3_levels.csv and equivalence.csv from S3 run ledgers.

    Args:
        run_dir: Run results directory.
        artifacts_dir: Run artifacts directory.
    """
    import csv
    import hashlib

    import pyarrow.parquet as pq

    ours_ledger_path = artifacts_dir / "s3" / "artifacts" / "s3" / "ours" / "run_0" / "ledger.parquet"
    direct_ledger_path = artifacts_dir / "s3" / "artifacts" / "s3" / "direct" / "run_0" / "ledger.parquet"

    if not ours_ledger_path.exists() or not direct_ledger_path.exists():
        return

    ours_table = pq.read_table(ours_ledger_path)
    direct_table = pq.read_table(direct_ledger_path)

    ours_rows = ours_table.to_pydict()
    direct_rows = direct_table.to_pydict()

    # Direct rows are aligned by position (one per sigma level)
    direct_shas = direct_rows["sha256"]

    # Write s3_levels.csv
    levels_path = run_dir / "s3_levels.csv"
    with levels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "sigma", "ksize", "output_id", "artifact"])
        for i, rid in enumerate(ours_rows["id"]):
            params = ours_rows["params"][i]
            import json

            p = json.loads(params)
            sigma = p["sigma_limit"][0]
            ksize = int(3.5 * sigma) * 2 + 1
            artifact = ours_rows["artifact"][i]
            writer.writerow([i, sigma, ksize, rid, artifact])

    # Write equivalence.csv
    eq_path = run_dir / "equivalence.csv"
    with eq_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "sigma", "ksize", "ours_output_id", "ours_sha256", "direct_sha256", "equal"])
        for i, rid in enumerate(ours_rows["id"]):
            params = ours_rows["params"][i]
            import json

            p = json.loads(params)
            sigma = p["sigma_limit"][0]
            ksize = int(3.5 * sigma) * 2 + 1
            artifact = ours_rows["artifact"][i]
            # Compute sha256 of ours payload
            payloads_dir = artifacts_dir / "s3" / "artifacts" / "s3" / "ours" / "run_0" / "payloads"
            ours_hash = hashlib.sha256((payloads_dir / artifact).read_bytes()).hexdigest()
            d_sha = direct_shas[i] if i < len(direct_shas) else ""
            equal = "true" if ours_hash == d_sha else "false"
            writer.writerow([i, sigma, ksize, rid, ours_hash, d_sha, equal])


def _load_b1_attr_rows(artifacts_dir: Path, limit: int = 100) -> list[dict[str, Any]] | None:
    """Load a small S1 subset with resolved image paths.

    Args:
        artifacts_dir: Run artifacts directory.
        limit: Maximum number of manifest rows to load.

    Returns:
        Manifest rows with resolved image paths, or None when inputs are missing.
    """
    import pyarrow.parquet as pq

    manifest_path = artifacts_dir / "inputs" / "manifest.parquet"
    if not manifest_path.exists():
        return None
    rows = pq.read_table(manifest_path).to_pylist()[:limit]
    images_dir = artifacts_dir / "inputs" / "images"
    for row in rows:
        row["path"] = str(images_dir / row["path"])
    return rows


def _load_b2_attr_rows(artifacts_dir: Path, suites: list[str], limit: int = 1) -> list[dict[str, Any]] | None:
    """Load comma10k rows with the image prefix resolved from the job config.

    Args:
        artifacts_dir: Run artifacts directory.
        suites: Suites that ran (s2 or s3 carries the resolved prefix).
        limit: Maximum number of rows to load.

    Returns:
        Manifest rows with resolved image paths, or None when inputs are missing.
    """
    import pyarrow.parquet as pq
    import yaml

    comma_path = artifacts_dir / "inputs" / "comma10k.parquet"
    if not comma_path.exists():
        return None
    rows = pq.read_table(comma_path).to_pylist()[:limit]

    prefix = "examples/data/comma10k_sample/imgs"
    for suite in ("s2", "s3"):
        config_path = artifacts_dir / suite / "configs" / f"{suite}.yaml"
        if suite not in suites or not config_path.exists():
            continue
        loaders = yaml.safe_load(config_path.read_text()).get("dataloaders", {}).get("loaders", [{}])
        sample = loaders[0].get("sample_path", [{}])[0] if loaders else {}
        prefix = sample.get("prefix") or prefix
        break

    for row in rows:
        row["path"] = str(Path(prefix) / row["path"])
    return rows


def _make_image_batch(image: Any, row_id: str) -> Any:
    """Wrap one decoded image in the image package's batch type.

    Args:
        image: One decoded ``(H, W, C)`` uint8 image.
        row_id: Parent row identifier.

    Returns:
        An ``ImageBatch`` of one row.
    """
    import pyarrow as pa

    from kcai_data_sampling_images.api.selection import ImageBatch

    return ImageBatch(
        name="attribution",
        dataset="attribution",
        ids=[row_id],
        columns=pa.table({"id": [row_id]}),
        data=image[None],
    )


def _build_ours_transform(entry: dict[str, Any]) -> Any:
    """Build the workspace-side transformation for one job entry.

    Deliberately imported lazily so a run without ``--with-attribution`` never
    imports the albumentations package through this module.

    Args:
        entry: A job-config transformation entry (type, seed, p, parameters).

    Returns:
        An ``AlbumentationsUnary`` instance built from the entry.
    """
    from kcai_data_sampling_albumentation.api.transformations.gaussian_blur import GaussianBlur
    from kcai_data_sampling_albumentation.api.transformations.random_brightness_contrast import (
        RandomBrightnessContrast,
    )

    classes = {
        "gaussian_blur": GaussianBlur,
        "random_brightness_contrast": RandomBrightnessContrast,
    }
    return classes[entry["type"]](entry)


def _attr_m3(func, *args) -> dict[str, Any]:
    """Profile one direct-side function with cProfile.

    Args:
        func: Function to profile.
        *args: Positional arguments for ``func``.

    Returns:
        A dict with the profile text.
    """
    from benchmark.attribution import run_m3_profiler

    return {"profile_text": run_m3_profiler(func, *args)["profile_text"]}


def _attr_m3_ours(entry: dict[str, Any], image: Any, row_id: str, calls: int = 30) -> dict[str, Any]:
    """Profile repeated workspace-side ``transform`` calls with cProfile.

    Args:
        entry: A job-config transformation entry.
        image: One decoded image.
        row_id: Parent row identifier.
        calls: Number of repeated transforms to profile.

    Returns:
        A dict with the call count and the profile text.
    """
    from benchmark.attribution import run_m3_profiler

    transform = _build_ours_transform(entry)
    batch = _make_image_batch(image, row_id)
    profile_text = run_m3_profiler(lambda: [transform.transform(batch, (0.0, 255.0)) for _ in range(calls)])[
        "profile_text"
    ]
    return {"calls": calls, "profile_text": profile_text}


def _attr_m5_ours(entry: dict[str, Any], image: Any, row_id: str) -> dict[str, float]:
    """Measure the workspace contract layer around one upstream transform.

    ``apply`` measures the engine's per-row work (reseeding, alpha split,
    upstream call, shape restore); ``transform`` adds the contract moves
    (``rngs()``, shape check, ``fit_to_range``, ``Output`` construction).

    Args:
        entry: A job-config transformation entry.
        image: One decoded ``(H, W, C)`` uint8 image.
        row_id: Parent row identifier.

    Returns:
        Median ms per call for rngs, apply and transform, plus the contract overhead.
    """
    from benchmark.attribution import run_m5_microbench

    transform = _build_ours_transform(entry)
    batch = _make_image_batch(image, row_id)
    rngs = transform.rngs(batch.ids)

    rngs_ms = run_m5_microbench(lambda: transform.rngs(batch.ids), iterations=200)["ms_per_call"]
    apply_ms = run_m5_microbench(lambda: transform.apply(batch.data, rngs), iterations=100)["ms_per_call"]
    transform_ms = run_m5_microbench(lambda: transform.transform(batch, (0.0, 255.0)), iterations=100)["ms_per_call"]
    overhead_ms = transform_ms - apply_ms
    return {
        "rngs_ms": rngs_ms,
        "apply_ms": apply_ms,
        "transform_ms": transform_ms,
        "contract_overhead_ms": overhead_ms,
        "contract_overhead_pct": (overhead_ms / transform_ms * 100.0) if transform_ms else 0.0,
    }


def _attr_m5_direct(transform: Any, image_rgb: Any) -> dict[str, float]:
    """Measure one upstream transform: kernel-only versus the full call.

    Upstream's ``apply`` needs the parameters that ``get_params`` and
    ``get_params_dependent_on_data`` normally draw, so they are sampled once
    and fixed for the kernel-only measurement.

    Args:
        transform: An upstream albumentations transform.
        image_rgb: One decoded ``(H, W, 3)`` uint8 image.

    Returns:
        Median ms per call for the kernel and the full transform call.
    """
    from benchmark.attribution import run_m5_microbench

    params = dict(transform.get_params())
    params.update(transform.get_params_dependent_on_data({}, {"image": image_rgb}))
    return {
        "kernel_ms": run_m5_microbench(lambda: transform.apply(image_rgb, **params), iterations=100)["ms_per_call"],
        "transform_ms": run_m5_microbench(lambda: transform(image=image_rgb), iterations=100)["ms_per_call"],
    }


def _attr_safe(store: dict[str, Any], key: str, func) -> None:
    """Run one attribution measurement, recording failures instead of raising.

    Attribution is optional post-gate instrumentation; a failure in one metric
    must not discard a completed run's headline outputs.

    Args:
        store: The section dict receiving the result.
        key: Metric name.
        func: Zero-argument callable producing the metric.
    """
    import sys

    try:
        store[key] = func()
    except Exception as exc:  # noqa: BLE001 - optional metric, must not kill the run
        print(f"attribution {key} failed: {exc}", file=sys.stderr)
        store[key] = {"error": str(exc)}


def _attr_b1_full_job(attr: dict[str, Any], artifacts_dir: Path) -> None:
    """Profile the full ours-side S1 job at every batch size (not just transform path).

    Args:
        attr: The attribution document, with 'm3' section.
        artifacts_dir: Run artifacts directory.
    """
    from benchmark.attribution import run_m3_full_job

    suite_dir = artifacts_dir / "s1"
    configs_dir = suite_dir / "configs"
    if not configs_dir.exists():
        return

    run_level = artifact_dir_for_b1_suite(suite_dir)

    for cfg_path in sorted(configs_dir.glob("b*.yaml")):
        cell_name = cfg_path.stem  # e.g. "s1", "b5", "b10", "b20"
        key = f"ours_b1_full_job_{cell_name}"

        _clean_job_outputs(run_level)

        def _run(cp=cfg_path) -> dict[str, Any]:
            return run_m3_full_job(cp)

        _attr_safe(attr["m3"], key, _run)


def artifact_dir_for_b1_suite(suite_dir: Path) -> Path:
    """Get the run-level directory where the job writes outputs.

    Args:
        suite_dir: Suite artifacts directory.

    Returns:
        Run-level directory.
    """
    # artifacts_dir / s1 -> parents[0] is s1, parents[1] is artifacts_dir
    return suite_dir.parent


def _clean_job_outputs(run_level: Path) -> None:
    """Clean previous job outputs for profiling.

    Args:
        run_level: Directory where job writes ledger/payloads.
    """
    import shutil

    ledger_path = run_level / "ledger.parquet"
    payloads_path = run_level / "payloads"
    if ledger_path.exists():
        ledger_path.unlink()
    if payloads_path.exists():
        shutil.rmtree(payloads_path)
    payloads_path.mkdir(parents=True, exist_ok=True)


def _attribution_b1(attr: dict[str, Any], artifacts_dir: Path, payloads_dir: Path) -> None:
    """Run S1 attribution (direct profile plus workspace/native microbench).

    Args:
        attr: The attribution document, with 'm3' and 'm5' sections.
        artifacts_dir: Run artifacts directory.
        payloads_dir: Destination directory for profiled payload writes.
    """
    import albumentations

    from benchmark.direct import _run_b1
    from benchmark.parity import decode_batch
    from benchmark.scenarios import S1_METHOD

    rows = _load_b1_attr_rows(artifacts_dir)
    if rows is None:
        return

    config = {"compute": {"load_batch_size": 1}}
    _attr_safe(attr["m3"], "direct_b1", lambda: _attr_m3(_run_b1, rows, config, payloads_dir, []))

    _, batch = decode_batch(rows[:1])
    image = batch[0]
    row_id = str(rows[0]["id"])
    entry = {"name": S1_METHOD["type"], **S1_METHOD}
    _attr_safe(attr["m5"], "b1_random_brightness_contrast", lambda: _attr_m5_ours(entry, image, row_id))
    _attr_safe(attr["m3"], "ours_b1", lambda: _attr_m3_ours(entry, image, row_id))

    direct = albumentations.RandomBrightnessContrast(
        brightness_limit=S1_METHOD["brightness_limit"],
        contrast_limit=S1_METHOD["contrast_limit"],
        p=S1_METHOD["p"],
    )
    _attr_safe(attr["m5"], "b1_random_brightness_contrast_direct", lambda: _attr_m5_direct(direct, image[..., :3]))


def _attribution_b2(attr: dict[str, Any], artifacts_dir: Path, suites: list[str], payloads_dir: Path) -> None:
    """Run S2/S3 attribution (direct profiles plus workspace/native microbench).

    Args:
        attr: The attribution document, with 'm3' and 'm5' sections.
        artifacts_dir: Run artifacts directory.
        suites: Suites that ran.
        payloads_dir: Destination directory for profiled payload writes.
    """
    import albumentations

    from benchmark.direct import _run_s2, _run_s3
    from benchmark.parity import decode_batch
    from benchmark.scenarios import s3_entries

    rows = _load_b2_attr_rows(artifacts_dir, suites)
    if rows is None:
        return

    if "s2" in suites:
        _attr_safe(attr["m3"], "direct_b2a", lambda: _attr_m3(_run_s2, rows, payloads_dir, []))
    if "s3" in suites:
        _attr_safe(attr["m3"], "direct_b2b", lambda: _attr_m3(_run_s3, rows, payloads_dir, []))

    _, batch = decode_batch(rows[:1])
    image = batch[0]
    row_id = str(rows[0]["id"])
    entry = s3_entries()[0]
    sigma = entry["sigma_limit"][0]

    def measure_ours() -> dict[str, float]:
        result = _attr_m5_ours(entry, image, row_id)
        result["sigma"] = sigma
        return result

    _attr_safe(attr["m5"], "s3_gaussian_blur", measure_ours)
    _attr_safe(attr["m3"], "ours_b2b", lambda: _attr_m3_ours(entry, image, row_id))

    direct = albumentations.GaussianBlur(sigma_limit=(sigma, sigma), p=1.0)
    _attr_safe(attr["m5"], "s3_gaussian_blur_direct", lambda: _attr_m5_direct(direct, image[..., :3]))


def _run_attribution(run_dir: Path, artifacts_dir: Path, suites: list[str]) -> None:
    """Run M3 (cProfile) and M5 (microbench) attribution and write attribution.json.

    Args:
        run_dir: Run results directory.
        artifacts_dir: Run artifacts directory.
        suites: List of suites that ran.
    """
    from benchmark.attribution import write_attribution

    attr: dict[str, Any] = {"m3": {}, "m5": {}}
    payloads_dir = run_dir / "attr_payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)

    if "s1" in suites:
        _attribution_b1(attr, artifacts_dir, payloads_dir)
        _attr_b1_full_job(attr, artifacts_dir)
    if "s2" in suites or "s3" in suites:
        _attribution_b2(attr, artifacts_dir, suites, payloads_dir)

    write_attribution(run_dir / "attribution.json", attr)


def _finalize_run(
    run_dir: Path,
    outputs_root: Path,
    artifacts_dir: Path,
    report_mode: str,
    suites: list[str],
    all_records: list[dict[str, Any]],
) -> None:
    """Write final outputs: timing/memory CSVs, report, and index entry.

    Args:
        run_dir: Run results directory.
        outputs_root: Root outputs directory.
        report_mode: Report mode ('timing' or 'all').
        suites: List of suite names.
        all_records: All run records.
    """
    _write_timing_csvs(run_dir, suites, all_records)
    _write_memory_csvs(run_dir, suites, all_records)

    if "s3" in suites:
        _write_s3_files(run_dir, artifacts_dir)

    # Write report
    report.write_report(run_dir, mode=report_mode)

    # Append index.csv
    artifacts_dir = outputs_root / "artifacts" / run_dir.name
    artifacts_size = sum(f.stat().st_size for f in artifacts_dir.rglob("*") if f.is_file())
    index_path = outputs_root / "index.csv"
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w") as f:
            f.write("run_id,report_mode,artifacts_size_bytes\n")
    with index_path.open("a") as f:
        f.write(f"{run_dir.name},{report_mode},{artifacts_size}\n")


def orchestrate(
    outputs_root: Path,
    quick: bool = False,
    only: str | None = None,
    report_mode: str = "timing",
    with_attribution: bool = False,
) -> Path:
    """Run the full benchmark.

    Args:
        outputs_root: Root directory for outputs (e.g., benchmark/outputs).
        quick: If True, use fewer repeats/warmups for dev mode.
        only: If set, run only this suite ('s1', 's2', 's3').
        report_mode: 'timing' or 'all'.
        with_attribution: If True, run M3/M5 attribution.

    Returns:
        The run directory path.
    """
    run_dir, artifacts_dir, suites, warmups, repeats = _setup_run_dir(outputs_root, quick, only, with_attribution)

    # Run each suite
    all_records: list[dict[str, Any]] = []
    for suite in suites:
        suite_dir = artifacts_dir / suite
        suite_dir.mkdir(parents=True, exist_ok=True)

        if suite == "s1":
            records = _run_s1_suite(suite_dir, warmups, repeats)
        elif suite == "s2":
            records = _run_s2_suite(suite_dir, warmups, repeats)
        elif suite == "s3":
            records = _run_s3_suite(suite_dir, warmups, repeats)
        else:
            continue

        all_records.extend(records)

    _finalize_run(run_dir, outputs_root, artifacts_dir, report_mode, suites, all_records)

    # Optional post-gate instrumentation: run it after the headline outputs are
    # written so a profiling failure cannot discard a completed run.
    if with_attribution:
        _run_attribution(run_dir, artifacts_dir, suites)

    return run_dir
