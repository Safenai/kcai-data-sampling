from __future__ import annotations
import os
import io
import json
import uuid
import hashlib
import random
import pathlib
import inspect
import base64
import time
from typing import Any, Dict, List, Optional, Tuple
import math
import shutil

import numpy as np
import pandas as pd
from PIL import Image

import yaml
import albumentations as A
import cv2
import pyarrow as pa
import pyarrow.parquet as pq


try:
    from albumentations.core.utils import set_seed as _albumentations_set_seed
except Exception:
    def _albumentations_set_seed(seed: int):
        # fallback : on s'appuie sur random.seed et np.random.seed (si version inf albu à 1.3)
        return
# contrôle randomisation
def _seed32(*parts, base_seed: int = 0) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode('utf-8'))
        h.update(b'|')
    return (int.from_bytes(h.digest(), 'little') ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF

def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    _albumentations_set_seed(seed)


# exemple de transformation custom
class ColorTemperatureShift(A.ImageOnlyTransform):
    """Chauffe/refroidit l'image en décalant R/B."""
    def __init__(self, delta: int = 0, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.delta = int(delta)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        out = img.astype(np.int16)
        out[..., 0] = np.clip(out[..., 0] + self.delta, 0, 255)  # R
        out[..., 2] = np.clip(out[..., 2] - self.delta, 0, 255)  # B
        return out.astype(np.uint8)

CUSTOM_TRANSFORMS = {
    "ColorTemperatureShift": ColorTemperatureShift,
}

# utils 
def _ensure_dir(path: str):
    if path:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def _np_from_pil(img: Image.Image) -> np.ndarray:
    return np.array(img)

def _load_image_from_row(
    row: pd.Series,
    image_source: str,
    col_image_bytes: Optional[str],
    col_file_path: Optional[str],
    base_dir: Optional[str] = None,
) -> np.ndarray:
    if image_source == "image_bytes":
        assert col_image_bytes and col_image_bytes in row, f"Colonne '{col_image_bytes}' absente"
        b = row[col_image_bytes]
        if isinstance(b, str):  
            try:
                b = b.encode("latin1")
            except Exception:
                try:
                    b = base64.b64decode(b)
                except Exception as e:
                    raise ValueError("image_bytes: ni latin1 ni base64.") from e
        img = _pil_from_bytes(b)
        return _np_from_pil(img)

    assert col_file_path and col_file_path in row, f"Colonne '{col_file_path}' absente"
    fp = str(row[col_file_path])
    if base_dir and not os.path.isabs(fp):
        fp = os.path.join(base_dir, fp)
    arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Impossible de lire l'image: {fp}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr

def _encode_image(arr_rgb: np.ndarray, ext: str = "png") -> bytes:
    if ext.lower() in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext.lower() in ("tif", "tiff"):
        ok, buf = cv2.imencode(".tif", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    else:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Échec encodage image")
    return buf.tobytes()

# utils paramètres pour les transformations
def _resolve_transform_class(name: str):
    if name in CUSTOM_TRANSFORMS:
        return CUSTOM_TRANSFORMS[name]
    return getattr(A, name)

def _coerce_pair_param(param_name: str, value):
    if isinstance(param_name, str) and (param_name.endswith("_range") or param_name.endswith("_limit")):
        if isinstance(value, (int, float)):
            return (value, value)
        if isinstance(value, list):
            if len(value) == 2:
                return (value[0], value[1])
            if len(value) == 1:
                return (value[0], value[0])
    return value

# 
def _normalize_params_by_version(name: str, params: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    p = dict(params)
    p.pop("level", None)

    if name == "RandomFog":
        if "fog_coef_range" in allowed and ("fog_coef_lower" in p or "fog_coef_upper" in p):
            low = p.pop("fog_coef_lower", p.get("fog_coef_upper", 0.3))
            up = p.pop("fog_coef_upper", p.get("fog_coef_lower", 1.0))
            p["fog_coef_range"] = (low, up)

    if name == "RandomSunFlare":
        if "angle_range" in allowed and ("angle_lower" in p or "angle_upper" in p):
            p["angle_range"] = (p.pop("angle_lower", 0.0), p.pop("angle_upper", 1.0))
        if "num_flare_circles_range" in allowed and ("num_flare_circles_lower" in p or "num_flare_circles_upper" in p):
            p["num_flare_circles_range"] = (p.pop("num_flare_circles_lower", 6), p.pop("num_flare_circles_upper", 10))

    if name == "RandomShadow":
        if "num_shadows_limit" in allowed and ("num_shadows_lower" in p or "num_shadows_upper" in p):
            low = p.pop("num_shadows_lower", 1)
            up = p.pop("num_shadows_upper", max(1, low))
            p["num_shadows_limit"] = (low, up)

    if name == "RandomRain":
        if "slant_range" in allowed and ("slant_lower" in p or "slant_upper" in p):
            p["slant_range"] = (p.pop("slant_lower", -10), p.pop("slant_upper", 10))

    if name == "RandomSnow":
        if "snow_point_range" in allowed and ("snow_point_lower" in p or "snow_point_upper" in p):
            p["snow_point_range"] = (p.pop("snow_point_lower", 0.1), p.pop("snow_point_upper", 0.3))

    if name == "CoarseDropout":
        if "num_holes_range" in allowed and ("min_holes" in p or "max_holes" in p):
            lo = p.pop("min_holes", None); hi = p.pop("max_holes", None)
            if lo is None and hi is not None: lo = hi
            if hi is None and lo is not None: hi = lo
            if lo is not None and hi is not None:
                p["num_holes_range"] = (int(lo), int(hi))
        if "hole_height_range" in allowed and ("min_height" in p or "max_height" in p):
            lo = p.pop("min_height", None); hi = p.pop("max_height", None)
            if lo is None and hi is not None: lo = hi
            if hi is None and lo is not None: hi = lo
            if lo is not None and hi is not None:
                p["hole_height_range"] = (int(lo), int(hi))
        if "hole_width_range" in allowed and ("min_width" in p or "max_width" in p):
            lo = p.pop("min_width", None); hi = p.pop("max_width", None)
            if lo is not None and hi is not None:
                p["hole_width_range"] = (int(lo), int(hi))
        if "fill" in allowed and "fill_value" in p and "fill" not in p:
            p["fill"] = p.pop("fill_value")
        if "fill_mask" in allowed and "mask_fill_value" in p and "fill_mask" not in p:
            p["fill_mask"] = p.pop("mask_fill_value")

    for k in list(p.keys()):
        if isinstance(k, str) and (k.endswith("_range") or k.endswith("_limit")):
            p[k] = _coerce_pair_param(k, p[k])
    return p

def _sanitize_params(transform_cls, name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sig = inspect.signature(transform_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    normalized = _normalize_params_by_version(name, dict(params), allowed)
    clean = {k: v for k, v in normalized.items() if k in allowed}
    ignored = {k: v for k, v in normalized.items() if k not in allowed}
    return clean, ignored

def _build_transform_instance(name: str, params: Dict[str, Any]) -> Tuple[A.BasicTransform, Dict[str, Any], Dict[str, Any]]:
    cls = _resolve_transform_class(name)
    clean, ignored = _sanitize_params(cls, name, params)
    inst = cls(**clean)
    return inst, clean, ignored

# def _iter_variants(transform_cfg: Dict[str, Any]):
#     # (compat historique — non utilisé quand on active le sampling par image/transform unique)
#     name = transform_cfg["name"]
#     base_params = dict(transform_cfg.get("params", {}))

#     if "level_overrides" in transform_cfg:
#         for overrides in transform_cfg["level_overrides"]:
#             params = dict(base_params)
#             level = overrides.get("level", None)
#             for k, v in overrides.items():
#                 if k == "level": continue
#                 params[k] = _coerce_pair_param(k, v)
#             instance, effective, ignored = _build_transform_instance(name, params)
#             yield {"tname": name, "level": level, "params": effective, "ignored": ignored, "instance": instance}
#         return

#     levels = transform_cfg.get("levels")
#     level_param = transform_cfg.get("level_param")
#     if levels and level_param:
#         for lvl in levels:
#             params = dict(base_params)
#             params[level_param] = _coerce_pair_param(level_param, lvl)
#             instance, effective, ignored = _build_transform_instance(name, params)
#             yield {"tname": name, "level": lvl, "params": effective, "ignored": ignored, "instance": instance}
#         return

#     samples_per_image = int(transform_cfg.get("samples_per_image", 0))
#     if samples_per_image > 0:
#         for k in range(samples_per_image):
#             instance, effective, ignored = _build_transform_instance(name, base_params)
#             yield {"tname": name, "level": k, "params": effective, "ignored": ignored, "instance": instance}
#         return

#     instance, effective, ignored = _build_transform_instance(name, base_params)
#     yield {"tname": name, "level": None, "params": effective, "ignored": ignored, "instance": instance}

def _normalize_weights(ws: Optional[List[float]], n: int) -> Optional[List[float]]:
    if not ws:
        return None
    if len(ws) != n:
        return None
    w = np.array(ws, dtype=float)
    s = w.sum()
    if s <= 0:
        return None
    return (w / s).tolist()

def _pick_transform_for_image(transforms_cfg: List[Dict[str, Any]], pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # poids au niveau pipeline (liste) ou par transform via 'select_weight' (si one_transform_per_image: false)
    n = len(transforms_cfg)
    weights = pipeline_cfg.get("transform_weights")
    if not weights:
        weights = [t.get("select_weight") for t in transforms_cfg] if any("select_weight" in t for t in transforms_cfg) else None
    p = _normalize_weights(weights, n)
    idx = int(np.random.choice(n, p=p))
    return transforms_cfg[idx]

def _sample_uniform_level(t_cfg: Dict[str, Any]) -> Optional[float]:
    sp = t_cfg.get("sampling")
    if sp and sp.get("mode", "uniform") == "uniform":
        lo, hi = sp.get("range", [0.0, 0.0])
        lo, hi = float(lo), float(hi)
        L = float(np.random.uniform(lo, hi))
        rd = sp.get("round")
        if rd is not None:
            L = round(L, int(rd))
        return L

    # fallback: si pas de sampling → si 'levels' existe, on choisit 1 niveau au hasard
    if "levels" in t_cfg and t_cfg["levels"]:
        return float(np.random.choice(t_cfg["levels"]))

    return None


class LevelMonitor: # monitoring de traitmenet
    def __init__(self, every: int = 0, log_on_start: bool = True, log_on_finish: bool = True):
        self.every = int(every)
        self.log_on_start = bool(log_on_start)
        self.log_on_finish = bool(log_on_finish)
        self.state: Dict[tuple, Dict[str, Any]] = {}

    def _fmt_key(self, tid, tname, level):
        lvl = "none" if level is None else level
        return f"id={tid} | name={tname} | level={lvl}"

    def start_if_needed(self, tid, tname, level):
        key = (tid, tname, level)
        if key not in self.state:
            self.state[key] = {"n": 0, "t0": time.time(), "sources": set()}
            if self.log_on_start:
                print(f"[start] {self._fmt_key(tid, tname, level)}")

    def tick(self, tid, tname, level, parent_id: str):
        key = (tid, tname, level)
        s = self.state[key]
        s["n"] += 1
        if parent_id is not None:
            s["sources"].add(parent_id)
        if self.every > 0 and (s["n"] % self.every == 0):
            elapsed = max(1e-6, time.time() - s["t0"])
            rate = s["n"] / elapsed
            print(f"[proc]  {self._fmt_key(tid, tname, level)}: n={s['n']} | sources={len(s['sources'])} | {elapsed:.1f}s | ~{rate:.1f} img/s")

    def finish_all(self):
        if not self.log_on_finish:
            return
        print("\n=== résumé par type/niveau ===")
        for (tid, tname, level), s in sorted(self.state.items(), key=lambda kv: (kv[0][0], str(kv[0][2]))):
            elapsed = max(1e-6, time.time() - s["t0"])
            rate = s["n"] / elapsed
            print(f"[done]  id={tid:<2} name={tname:<18} level={('none' if level is None else level):<6} "
                  f"n={s['n']:<6} sources={len(s['sources']):<5} time={elapsed:.1f}s rate={rate:.1f} img/s")

# ==================== Runner ====================
def _stable_uid(base_id: str, tname: str, level: Any, params: Dict[str, Any]) -> str:
    h = hashlib.sha1()
    payload = json.dumps({"base": base_id, "tname": tname, "level": level, "params": params}, sort_keys=True).encode("utf-8")
    h.update(payload)
    return h.hexdigest()[:10]

class SafeDict(dict):
    def __missing__(self, key):
        return ""

def _write_table_single(df: pd.DataFrame, table_out_path: str, out_type: str, compression: Optional[str]):
    _ensure_dir(os.path.dirname(table_out_path))
    if out_type == "csv":
        df.to_csv(table_out_path, index=False)
    elif out_type == "parquet":
        df.to_parquet(table_out_path, index=False, compression=compression)
    else:
        raise ValueError(f"Unsupported table output type: {out_type}")

def _clear_partition_dir(root_path: str, partition_cols: List[str], key_values: List[str]):
    path = root_path
    for col, val in zip(partition_cols, key_values):
        path = os.path.join(path, f"{col}={val}")
    shutil.rmtree(path, ignore_errors=True)

def _write_parquet_dataset(
    df: pd.DataFrame,
    root_path: str,
    partition_cols: List[str],
    compression: Optional[str],
    existing_data_behavior: str = "overwrite_or_ignore",
    max_rows_per_file: Optional[int] = None,
    files_per_partition_min: Optional[int] = None,
    max_rows_per_group: Optional[int] = None,
):
    if df.empty:
        _ensure_dir(root_path)
        return

    df2 = df.copy()
    for col in partition_cols:
        if col not in df2.columns:
            raise ValueError(f"partition col '{col}' absente de la table")
    if "transform_level" in partition_cols:
        df2["transform_level"] = df2["transform_level"].apply(lambda v: "none" if pd.isna(v) else str(v))
    for col in partition_cols:
        if col != "transform_level":
            df2[col] = df2[col].astype(str)

    _ensure_dir(root_path)
    mrf = int(max_rows_per_file) if max_rows_per_file is not None else None
    mrg = int(max_rows_per_group) if max_rows_per_group is not None else None
    if mrf is not None and (mrg is None or mrg > mrf):
        mrg = mrf

    if not files_per_partition_min or int(files_per_partition_min) <= 1:
        table = pa.Table.from_pandas(df2, preserve_index=False)
        kwargs = {}
        if mrf is not None:
            kwargs["max_rows_per_file"] = mrf
        if mrg is not None:
            kwargs["max_rows_per_group"] = mrg
            kwargs["row_group_size"] = mrg
        pq.write_to_dataset(
            table,
            root_path=root_path,
            partition_cols=partition_cols,
            compression=compression if compression else None,
            existing_data_behavior=existing_data_behavior,
            **kwargs,
        )
        return

    grouped = df2.groupby(partition_cols, dropna=False, sort=False)
    for key_vals, part in grouped:
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_vals_str = [str(v) for v in key_vals]

        if existing_data_behavior == "delete_matching":
            _clear_partition_dir(root_path, partition_cols, key_vals_str)

        nrows = len(part)
        by_size = math.ceil(nrows / mrf) if mrf else 1
        n_files = max(int(files_per_partition_min), by_size)
        rows_per_shard = max(1, math.ceil(nrows / n_files))

        for i in range(0, nrows, rows_per_shard):
            shard = part.iloc[i:i+rows_per_shard]
            table = pa.Table.from_pandas(shard, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=root_path,
                partition_cols=partition_cols,
                compression=compression if compression else None,
                existing_data_behavior="overwrite_or_ignore",
            )

def run_pipeline(cfg: Dict[str, Any]) -> str:
    ap = cfg["augment_pipeline"]

    # loading des données
    dl = ap["dataloader"]
    input_type = dl["type"]
    input_path = dl["path"]
    batch_size = int(dl.get("batch_size", 10_000))
    columns = dl["columns"]
    image_source = columns.get("image_source", "image_bytes")
    col_bytes = columns.get("image_bytes")
    col_path = columns.get("file_path")
    base_dir = dl.get("base_dir")

    # outputs - parquet et partionning
    outputs = ap["outputs"]
    img_out = outputs["image_output"]
    file_ext = img_out.get("file_ext", "png")

    table_out = outputs["table_output"]
    table_out_type = table_out["type"]
    table_out_path = table_out["path"]
    compression = table_out.get("compression", None)
    dataset_mode = bool(table_out.get("dataset", False)) or bool(table_out.get("partition_by"))
    partition_cols = list(table_out.get("partition_by", []))
    existing_behavior = table_out.get("existing_data_behavior", "overwrite_or_ignore")
    max_rows_per_file = table_out.get("max_rows_per_file")
    files_per_partition_min = table_out.get("files_per_partition_min")
    max_rows_per_group = table_out.get("max_rows_per_group")

    # metrics
    metrics_out = outputs.get("metrics_output", None)
    ordered_front = outputs["columns"]["ordered_front"]
    include_src = outputs["columns"].get("include_source_columns", False)

    # run 
    run_opts = ap.get("run", {})
    seed = int(run_opts.get("seed", 123))
    deterministic_ids = bool(run_opts.get("deterministic_ids", False))
    id_pattern = run_opts.get("id_pattern", "{base}__{tname}__{level}__{uid}")
    output_sample_type = run_opts.get("output_sample_type", "perturbation")

    # monitor
    monitor_cfg = ap.get("monitor", {})
    per_level_every = int(monitor_cfg.get("log_every_per_level", 0))
    log_on_start = bool(monitor_cfg.get("log_on_start", True))
    log_on_finish = bool(monitor_cfg.get("log_on_finish", True))
    mon = LevelMonitor(per_level_every, log_on_start, log_on_finish)

    _seed_all(seed)

    # read
    if input_type == "csv":
        full_df = pd.read_csv(input_path)
    elif input_type == "parquet":
        full_df = pd.read_parquet(input_path)
    else:
        raise ValueError("'")

    pipelines = ap.get("augmentation_pipelines", [])
    if not pipelines:
        raise ValueError("aucune pipeline/transforms fournie")

    all_records: List[Dict[str, Any]] = []
    n = len(full_df)
    loaded_ok, loaded_fail = 0, 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = full_df.iloc[start:end]

        for _, row in batch.iterrows():
            try:
                arr = _load_image_from_row(row, image_source, col_bytes, col_path, base_dir=base_dir)
                loaded_ok += 1
            except Exception:
                loaded_fail += 1
                continue

            base_id = row.get(columns.get("sample_id", "sample_id"), f"row{start}")
            class_id = row.get(columns.get("class_id", "class_id"), None)
            class_name = row.get(columns.get("class_name", "class_name"), None)

            for pipe_idx, pipe_cfg in enumerate(pipelines):
                # on a la décision apply_prob, reseedée par image + pipeline
                apply_prob = float(pipe_cfg.get("apply_prob", 1.0))
                if apply_prob < 1.0:
                    _seed_all(_seed32(base_id, pipe_idx, "apply_prob", base_seed=seed))
                    if random.random() > apply_prob:
                        continue  # skip de manière déterministe

                # option de resizing 
                resize = pipe_cfg.get("resize")
                if resize:
                    H, W = resize
                    arr_in = A.Resize(height=H, width=W)(image=arr)["image"]
                else:
                    arr_in = arr

                transforms_cfg = pipe_cfg.get("transforms", [])
                if not transforms_cfg:
                    continue

                # Choix DU transform (1 seul) de façon déterministe
                one_transform = bool(pipe_cfg.get("one_transform_per_image", True))
                if one_transform:
                    _seed_all(_seed32(base_id, pipe_idx, "choose_transform", base_seed=seed))
                    chosen_t = _pick_transform_for_image(transforms_cfg, pipe_cfg)
                    t_cfgs = [chosen_t]
                else:
                    t_cfgs = transforms_cfg  # compat: plusieurs sorties (ancienne version)

                for t_cfg in t_cfgs:
                    transform_alias = t_cfg.get("alias")
                    transform_id_value = int(t_cfg.get("id", 0))
                    alb_name = t_cfg["name"]
                    level_param = t_cfg.get("level_param")

                    # tirage du niveau: reseed dédié pour le level
                    _seed_all(_seed32(base_id, transform_id_value, "level", base_seed=seed))
                    level = _sample_uniform_level(t_cfg)

                    params = dict(t_cfg.get("params", {}))
                    if level_param is not None and level is not None:
                        params[level_param] = _coerce_pair_param(level_param, level)

                    t, params_used, _ignored = _build_transform_instance(alb_name, params)
                    tname_display = transform_alias or alb_name

                    _seed_all(_seed32(base_id, transform_id_value, level, "apply", base_seed=seed))

                    mon.start_if_needed(transform_id_value, tname_display, level)
                    out_arr = A.Compose([t])(image=arr_in)["image"]
                    img_bytes = _encode_image(out_arr, file_ext)

                    # iD stable optionnel
                    uid = _stable_uid(base_id, tname_display, level, params_used) if deterministic_ids else str(uuid.uuid4())[:8]
                    fmt_vars = SafeDict({
                        "base": base_id,
                        "tname": tname_display,
                        "level": ("none" if level is None else str(level)),
                        "uid": uid
                    })
                    child_id = id_pattern.format_map(fmt_vars)

                    rec = {
                        # optionnel:  pour avoir un ID unique par sortie, utiliser "sample_id": child_id
                        "sample_id": base_id,
                        "sample_type": output_sample_type,
                        "class_id": class_id,
                        "class_name": class_name,
                        "transform_id": transform_id_value,
                        "transform_name": tname_display,
                        "transform_level": level,
                        "transform_params": json.dumps(params_used, ensure_ascii=False),
                        "image_bytes": img_bytes,
                    }
                    if include_src:
                        for c in full_df.columns:
                            if c not in rec:
                                rec[f"src__{c}"] = row[c]

                    all_records.append(rec)
                    mon.tick(transform_id_value, tname_display, level, base_id)

    out_df = pd.DataFrame(all_records)

    if not out_df.empty:
        front = ordered_front
        rest = [c for c in out_df.columns if c not in front]
        out_df = out_df[front + rest]

    print(f"Sources chargées: {loaded_ok}, échecs: {loaded_fail}")
    print(f"Augmentations générées: {len(out_df)}")

    if dataset_mode:
        _write_parquet_dataset(
            out_df,
            table_out_path,
            partition_cols or ["transform_name", "transform_level"],
            compression,
            existing_data_behavior=existing_behavior,
            max_rows_per_file=int(max_rows_per_file) if max_rows_per_file else None,
            files_per_partition_min=int(files_per_partition_min) if files_per_partition_min else None,
            max_rows_per_group=int(max_rows_per_group) if max_rows_per_group else None,
        )
        print(f" dataset Parquet partitionné -> {table_out_path}")
    else:
        _write_table_single(out_df, table_out_path, table_out_type, compression)
        print(f" {table_out_type} écrit -> {table_out_path}")

    if outputs.get("metrics_output"):
        metrics = outputs["metrics_output"]
        m_type = metrics.get("type", "parquet")
        m_path = metrics["path"]
        m_comp = metrics.get("compression")

        rows: List[Dict[str, Any]] = []
        if not out_df.empty:
            g = out_df.groupby(["transform_id", "transform_name"], dropna=False)
            for (tid, tname), df_g in g:
                rows.append({
                    "scope": "transform",
                    "transform_id": int(tid) if pd.notna(tid) else None,
                    "transform_name": tname,
                    "transform_level": None,
                    "n_images": int(len(df_g)),
                })
            g2 = out_df.groupby(["transform_id", "transform_name", "transform_level"], dropna=False)
            for (tid, tname, lvl), df_g in g2:
                rows.append({
                    "scope": "transform_level",
                    "transform_id": int(tid) if pd.notna(tid) else None,
                    "transform_name": tname,
                    "transform_level": lvl,
                    "n_images": int(len(df_g)),
                })
        mdf = pd.DataFrame(rows)
        _ensure_dir(os.path.dirname(m_path))
        if m_type == "csv":
            mdf.to_csv(m_path, index=False)
        else:
            mdf.to_parquet(m_path, index=False, compression=m_comp)
        print(f" métriques écrites -> {m_path}")

    mon.finish_all()
    return table_out_path
