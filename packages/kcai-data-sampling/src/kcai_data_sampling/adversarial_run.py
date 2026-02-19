# adversarial_runner.py
from __future__ import annotations
import os, io, json, uuid, hashlib, random, pathlib, base64, time, math, shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import yaml
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm

# attaques sous-jacentes (utilisés dans robust-ai) 
# CleverHans pour FGSM/PGD/CW

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method   # FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent # PGD
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2         # C&W (L2)
# AutoAttack / APGD
from pyautoattack import AutoAttack

# def _seed32(*parts, base_seed: int = 0) -> int:
#     h = hashlib.blake2b(digest_size=8)
#     for p in parts:
#         h.update(str(p).encode("utf-8")); h.update(b"|")
#     return (int.from_bytes(h.digest(), "little") ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF

def _ensure_dir(path: str):
    if path:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def _np_from_pil(img: Image.Image) -> np.ndarray:
    return np.array(img)

def _encode_image(arr_rgb: np.ndarray, ext: str = "png") -> bytes:
    if ext.lower() in ("jpg","jpeg"):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext.lower() in ("tif","tiff"):
        ok, buf = cv2.imencode(".tif", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    else:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
    if not ok: raise RuntimeError("Échec encodage image")
    return buf.tobytes()

def _load_image_from_row(row: pd.Series, image_source: str, col_image_bytes: Optional[str], col_file_path: Optional[str], base_dir: Optional[str]=None) -> np.ndarray:
    if image_source == "image_bytes":
        b = row[col_image_bytes]
        if isinstance(b, str):
            try: b = b.encode("latin1")
            except Exception: b = base64.b64decode(b)
        return _np_from_pil(_pil_from_bytes(b))
    fp = str(row[col_file_path])
    if base_dir and not os.path.isabs(fp): fp = os.path.join(base_dir, fp)
    arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if arr is None: raise FileNotFoundError(fp)
    if arr.ndim == 2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr

def _write_table_single(df: pd.DataFrame, table_out_path: str, out_type: str, compression: Optional[str]):
    _ensure_dir(os.path.dirname(table_out_path))
    if out_type == "csv": df.to_csv(table_out_path, index=False)
    elif out_type == "parquet": df.to_parquet(table_out_path, index=False, compression=compression)
    else: raise ValueError(f"Unsupported table output type: {out_type}")

def _clear_partition_dir(root_path: str, partition_cols: List[str], key_values: List[str]):
    path = root_path
    for col, val in zip(partition_cols, key_values):
        path = os.path.join(path, f"{col}={val}")
    shutil.rmtree(path, ignore_errors=True)

def _write_parquet_dataset(
    df: pd.DataFrame, root_path: str, partition_cols: List[str], compression: Optional[str],
    existing_data_behavior: str="overwrite_or_ignore", max_rows_per_file: Optional[int]=None,
    files_per_partition_min: Optional[int]=None, max_rows_per_group: Optional[int]=None,
):
    if df.empty:
        _ensure_dir(root_path); return
    df2 = df.copy()
    if "transform_level" in partition_cols:
        df2["transform_level"] = df2["transform_level"].apply(lambda v: "none" if pd.isna(v) else str(v))
    for col in partition_cols:
        if col != "transform_level":
            df2[col] = df2[col].astype(str)
    _ensure_dir(root_path)
    mrf = int(max_rows_per_file) if max_rows_per_file else None
    mrg = int(max_rows_per_group) if max_rows_per_group else None
    if mrf and (not mrg or mrg > mrf): mrg = mrf
    if not files_per_partition_min or int(files_per_partition_min) <= 1:
        table = pa.Table.from_pandas(df2, preserve_index=False)
        kwargs = {}
        if mrf: kwargs["max_rows_per_file"] = mrf
        if mrg: kwargs["max_rows_per_group"] = mrg; kwargs["row_group_size"] = mrg
        pq.write_to_dataset(table, root_path=root_path, partition_cols=partition_cols,
                            compression=compression if compression else None,
                            existing_data_behavior=existing_data_behavior, **kwargs)
        return
    grouped = df2.groupby(partition_cols, dropna=False, sort=False)
    for key_vals, part in grouped:
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
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
            pq.write_to_dataset(table, root_path=root_path, partition_cols=partition_cols,
                                compression=compression if compression else None,
                                existing_data_behavior="overwrite_or_ignore")

def _stable_uid(base_id: str, tname: str, level: Any, params: Dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"base": base_id, "tname": tname, "level": level, "params": params}, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:10]

class SafeDict(dict):
    def __missing__(self, key): return ""

# modèle & préproc 
def _to_tensor(img: np.ndarray, resize: Optional[Tuple[int,int]]=None) -> torch.Tensor:
    if resize: img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float()/255.0  # [0,1]
    t = t.permute(2,0,1)  # HWC->CHW
    return t

def _normalize(t: torch.Tensor, mean, std):
    mean = torch.tensor(mean, device=t.device).view(3,1,1)
    std  = torch.tensor(std, device=t.device).view(3,1,1)
    return (t - mean) / std

def _denormalize(t: torch.Tensor, mean, std):
    mean = torch.tensor(mean, device=t.device).view(3,1,1)
    std  = torch.tensor(std, device=t.device).view(3,1,1)
    return (t * std + mean)

def _clip01(t: torch.Tensor) -> torch.Tensor:
    return torch.clamp(t, 0.0, 1.0)

def _predict_logits(model, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)

#  adv attacks 
def run_fgsm(model, x, y, eps, clip_min=-1.0, clip_max=1.0):
    # CleverHans attend x normalisé comme utilisé au forward 
    def f(x_): return model(x_)
    x_adv = fast_gradient_method(model_fn=f, x=x, eps=eps, norm=np.inf, y=y, targeted=False, clip_min=clip_min, clip_max=clip_max)
    return x_adv

def run_pgd(model, x, y, eps, alpha, steps, rand_init=True, clip_min=-1.0, clip_max=1.0):
    def f(x_): return model(x_)
    x_adv = projected_gradient_descent(model_fn=f, x=x, eps=eps, eps_iter=alpha, nb_iter=steps,
                                       norm=np.inf, y=y, targeted=False, rand_init=rand_init,
                                       clip_min=clip_min, clip_max=clip_max)
    return x_adv

def run_cw(model, x, y, steps=1000, c=0.1, k=0.0, lr=5e-3, clip_min=-1.0, clip_max=1.0):
    # C&W L2 dans CleverHans; renvoie un tenseur adv
    # Get number of classes from model's final layer
    if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
        n_classes = model.fc.out_features
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        n_classes = model.classifier.out_features
    else:
        n_classes = 1000  # fallback
    
    x_adv = carlini_wagner_l2(model, x, n_classes=n_classes,
                              targeted=False, y=y, lr=lr, confidence=k, 
                              clip_min=clip_min, clip_max=clip_max,
                              max_iterations=steps, initial_const=c)
    return x_adv

def run_apgd(model, x, y, eps, n_restarts=1):
    # AutoAttack s'utilise en lot; on passe un wrapper
    # Utiliser le device des données d'entrée (x.device) au lieu de forcer CUDA
    device = x.device
    aa = AutoAttack(model, norm='Linf', eps=eps, version='custom', verbose=False, device=device)
    aa.attacks_to_run = ['apgd-ce']
    aa.n_restarts = int(n_restarts)
    x_adv = aa.run_standard_evaluation(x, y, bs=x.shape[0], return_labels=False)
    return x_adv

# runner principal 
def run_pipeline(cfg: Dict[str, Any]) -> str:
    ap = cfg["adversarial_pipeline"]

    # model & préproc
    mcfg = ap["model"]
    resize_hw = mcfg.get("resize")
    mean = mcfg.get("mean", [0.0,0.0,0.0])
    std  = mcfg.get("std",  [1.0,1.0,1.0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            _ = torch.tensor([1.0], device=device) + 1.0
            torch.cuda.synchronize()
        except Exception as e:
            print(f"CUDA non utilisable avec cette build ({str(e)[:100]}). Bascule sur CPU.")
            device = torch.device("cpu")
    ckpt = mcfg["ckpt_path"]
    model_name = os.path.basename(ckpt)  # nom du modèle
    # Ensure quantized backend is set on CPU BEFORE loading TorchScript
    if torch.device("cpu").type == 'cpu':
        supported_engines = torch.backends.quantized.supported_engines
        if 'fbgemm' in supported_engines:
            torch.backends.quantized.engine = 'fbgemm'
        elif 'qnnpack' in supported_engines:
            torch.backends.quantized.engine = 'qnnpack'
        # Note: we don't print here to keep logs concise

    # Try to load as TorchScript first (for .pt files), otherwise as regular checkpoint
    try:
        # Force TorchScript quantized models to CPU
        model = torch.jit.load(ckpt, map_location=torch.device("cpu"))
        print(f"✅ Modèle TorchScript chargé : {model_name} (CPU)")
        device = torch.device("cpu")
    except RuntimeError:
        # Not a TorchScript file, try regular checkpoint on CPU
        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)
        device = torch.device("cpu")
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Standard checkpoint format with model state dict
                import torchvision.models as models
                arch = checkpoint.get("arch", "resnet18")
                num_classes = checkpoint.get("num_classes", 1000)
                if arch == "resnet18":
                    model = models.resnet18(pretrained=False)
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                else:
                    raise ValueError(f"Unsupported architecture: {arch}")
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model = checkpoint
        else:
            model = checkpoint
        print(f"✅ Modèle checkpoint chargé : {model_name} (CPU)")

    if hasattr(model, "eval"): model.eval()
    model = model.to(device)

    # data - import
    dl = ap["dataloader"]
    full_df = pd.read_parquet(dl["path"]) if dl["type"]=="parquet" else pd.read_csv(dl["path"])
    cols = dl["columns"]
    image_source = cols.get("image_source", "image_bytes")
    col_bytes = cols.get("image_bytes"); col_path = cols.get("file_path")
    batch_size = int(dl.get("batch_size", 128))

    # data - output
    out = ap["outputs"]
    file_ext = out["image_output"].get("file_ext", "png")
    t_out = out["table_output"]; m_out = out.get("metrics_output")
    ordered_front = out["columns"]["ordered_front"]
    include_src = out["columns"].get("include_source_columns", False)

    # run opts
    run_opts = ap.get("run", {})
    seed = int(run_opts.get("seed", 123))
    deterministic_ids = bool(run_opts.get("deterministic_ids", False))
    id_pattern = run_opts.get("id_pattern", "{base}__{tname}__{level}__{uid}")
    output_sample_type = run_opts.get("output_sample_type", "adversarial")
    rng = np.random.RandomState(seed)

    attacks = ap["attacks"]
    all_records: List[Dict[str, Any]] = []  # Pour les métriques finales

    # Calcul des bornes de clipping dans l'espace normalisé
    # Si input_range est [0,1] et on applique (x - mean) / std :
    # clip_min = (0 - mean) / std, clip_max = (1 - mean) / std
    clip_min_normalized = (0.0 - np.array(mean)) / np.array(std)
    clip_max_normalized = (1.0 - np.array(mean)) / np.array(std)
    clip_min = float(clip_min_normalized.min())  # Prendre le min sur les 3 canaux
    clip_max = float(clip_max_normalized.max())  # Prendre le max sur les 3 canaux

    def _sample_level(a_cfg):
        sp = a_cfg.get("sampling")
        if not sp: return None
        lo, hi = map(float, sp["range"])
        val = float(rng.uniform(lo, hi))
        rd = sp.get("round"); 
        if rd is not None: val = round(val, int(rd))
        return val

    n = len(full_df)
    num_batches = (n + batch_size - 1) // batch_size
    total_operations = num_batches * len(attacks)

    print(f"\n{'='*60}")
    print(f" --- Starting adversarial pipeline ---")
    print(f" 📊 {n} images | {len(attacks)} attacks | {num_batches} batches")
    print(f" ⚙️  Total: {total_operations} operations")
    print(f"{'='*60}\n")

    pbar = tqdm(total=total_operations, desc="Global pipeline", unit="op", ncols=100)

    # loop on attacks (for incremental writing)
    for attack_idx, a_cfg in enumerate(attacks):
        alias = a_cfg.get("alias", a_cfg["name"])
        aid = int(a_cfg.get("id", 0))

        print(f"\n🎯 Attack {attack_idx+1}/{len(attacks)}: {alias} (id={aid})")

        records: List[Dict[str, Any]] = []  # Records for this attack

        # loop on batch for this attack
        for start in range(0, n, batch_size):
            part = full_df.iloc[start:start+batch_size]
            # prepare batch tensors
            imgs, meta = [], []
            for _, row in part.iterrows():
                try:
                    arr = _load_image_from_row(row, image_source, col_bytes, col_path)
                except Exception:
                    continue
                t = _to_tensor(arr, resize=resize_hw)
                imgs.append(t); meta.append(row)
            if not imgs: 
                pbar.update(1)
                continue
            x0 = torch.stack(imgs, dim=0).to(device)  # [B,3,H,W] in [0,1]
            x_n = _normalize(x0, mean, std)

            # reference labels (if class_id available); otherwise logits argmax of the model
            with torch.no_grad():
                logits = model(x_n)
            if "class_id" in cols and cols["class_id"] in part.columns and part[cols["class_id"]].notna().all():
                y = torch.tensor(part[cols["class_id"]].values, device=device).long()
            else:
                y = logits.argmax(1)
            
            level = _sample_level(a_cfg)
            params = dict(a_cfg.get("params", {}))
            lvl_param = a_cfg.get("level_param"); 
            if lvl_param is not None and level is not None: params[lvl_param] = level

            # update progress bar description
            batch_idx = start // batch_size + 1
            level_str = f"lvl={level:.3f}" if level is not None else "lvl=N/A"
            pbar.set_description(f"Batch {batch_idx}/{num_batches} | 🎯 {alias} ({level_str})")
            
            # prepare x for attack (normalized)
            x_in = x_n.detach()

            # dispatch attacks
            lib = a_cfg.get("lib", "cleverhans").lower()
            if alias == "fgsm" and lib == "cleverhans":
                eps = float(params["eps"])
                x_adv_n = run_fgsm(model, x_in, y, eps=eps, clip_min=clip_min, clip_max=clip_max)
            elif alias == "pgd" and lib == "cleverhans":
                eps = float(params["eps"])
                alpha = float(a_cfg.get("step_size_ratio", 0.25)) * eps
                steps = int(params.get("steps", 10))
                x_adv_n = run_pgd(model, x_in, y, eps=eps, alpha=alpha, steps=steps, rand_init=bool(params.get("rand_init", True)), clip_min=clip_min, clip_max=clip_max)
            elif alias == "cw" and lib == "cleverhans":
                steps = int(params.get("steps", 1000))
                c = float(params.get("c", 0.1))
                k = float(params.get("confidence", 0.0))
                lr_cw = float(params.get("lr", 5e-3))
                # C&W expects normalized input and works in the normalized space
                x_adv_n = run_cw(model, x_in, y, steps=steps, c=c, k=k, lr=lr_cw, clip_min=clip_min, clip_max=clip_max)
            elif alias.startswith("apgd") and lib == "pyautoattack":
                eps = float(params["eps"])
                n_restarts = int(params.get("n_restarts", 1))
                # AutoAttack returns already preprocessed pixels for the model → here it's in the normalized space
                x_adv_n = run_apgd(model, x_in, y, eps=eps, n_restarts=n_restarts)
            else:
                raise ValueError(f"Attack not supported: {alias} ({lib})")

            # denormalize to save PNG
            x_adv = _clip01(_denormalize(x_adv_n, mean, std))
            # NOTE: Inference removed - use kc_inference.py to add predictions

            # save lines
            for i in range(x_adv.shape[0]):
                arr = (x_adv[i].detach().cpu().permute(1,2,0).numpy()*255.0).round().astype(np.uint8)
                img_bytes = _encode_image(arr, file_ext)
                
                # Calculate hash of the adversarial image
                img_hash = hashlib.sha256(img_bytes).hexdigest()
                
                # Generate UUIDs
                sample_uuid_val = str(uuid.uuid4())
                parent_sample_uuid_val = meta[i].get(cols.get("sample_uuid", "sample_uuid"), str(uuid.uuid4()))
                
                base_id = meta[i].get(cols.get("sample_id","sample_id"), f"row{start+i}")
                params_used = {"alias": alias, **params}
                uid = _stable_uid(base_id, alias, level, params_used) if deterministic_ids else str(uuid.uuid4())[:8]
                child_id = (run_opts.get("id_pattern","{base}__{tname}__{level}__{uid}")).format_map(
                    SafeDict({"base": base_id, "tname": alias, "level": ("none" if level is None else str(level)), "uid": uid})
                )
                rec = {
                    "sample_uuid": sample_uuid_val,
                    "parent_sample_uuid": parent_sample_uuid_val,
                    "sample_id": base_id,
                    "sample_type": output_sample_type,
                    "class_id": int(meta[i].get(cols.get("class_id","class_id"), -1)) if cols.get("class_id") in meta[i] else None,
                    "class_name": meta[i].get(cols.get("class_name","class_name"), None),
                    "split": meta[i].get(cols.get("split","split"), None),
                    "transform_id": aid,
                    "transform_name": alias,
                    "transform_level": level,
                    "transform_params": json.dumps(params_used, ensure_ascii=False),
                    "image_bytes": img_bytes,
                    "image_hash": img_hash,
                }
                if include_src:
                    for c in full_df.columns:
                        if c not in rec: rec[f"src__{c}"] = meta[i].get(c, None)
                records.append(rec)
            
            # update progress bar after each batch
            pbar.update(1)
        
        # incremental writing after each attack
        if records:
            attack_df = pd.DataFrame(records)
            if not attack_df.empty:
                front = ordered_front
                rest = [c for c in attack_df.columns if c not in front]
                attack_df = attack_df[front + rest]
            
            # write data for this attack
            if out["table_output"].get("dataset", False) and out["table_output"].get("partition_by"):
                # Write as partitioned dataset
                _write_parquet_dataset(
                    attack_df,
                    out["table_output"]["path"],
                    out["table_output"].get("partition_by", ["transform_name","transform_level"]),
                    out["table_output"].get("compression"),
                    existing_data_behavior="overwrite_or_ignore",
                    max_rows_per_file=out["table_output"].get("max_rows_per_file"),
                    files_per_partition_min=out["table_output"].get("files_per_partition_min"),
                    max_rows_per_group=out["table_output"].get("max_rows_per_group"),
                )
                print(f"   ✅ {len(records)} échantillons écrits pour {alias} (partitioned)")
            else:
                # For single file mode, accumulate all records and write at the end
                print(f"   📦 {len(records)} échantillons accumulés pour {alias}")
            
            all_records.extend(records)
            
            # free memory
            del records, attack_df
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    pbar.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Pipeline terminated | {len(all_records)} samples generated")
    print(f"{'='*60}\n")

    # The data has already been written incrementally (if dataset mode)
    # For single file mode, write all accumulated records now
    out_df = pd.DataFrame(all_records)
    
    if not out["table_output"].get("dataset", False) or not out["table_output"].get("partition_by"):
        # Single file mode: write all records to one file
        if not out_df.empty:
            front = out["columns"]["ordered_front"]
            rest = [c for c in out_df.columns if c not in front]
            out_df = out_df[front + rest]
            
            output_path = out["table_output"]["path"]
            compression = out["table_output"].get("compression", "snappy")
            _write_table_single(out_df, output_path, "parquet", compression)
            print(f"✅ Wrote {len(out_df)} records to single file: {output_path}")
        else:
            print("⚠️  No records to write")

    # aggregated metrics (by attack/level)
    # NOTE: success_rate removed - use kc_inference.py to calculate after adding predictions
    if out.get("metrics_output"):
        rows = []
        if not out_df.empty:
            g = out_df.groupby(["transform_id","transform_name"], dropna=False)
            for (tid, tname), df_g in g:
                rows.append({
                    "scope": "attack",
                    "transform_id": int(tid) if pd.notna(tid) else None,
                    "transform_name": tname,
                    "transform_level": None,
                    "n_images": int(len(df_g)),
                })
            g2 = out_df.groupby(["transform_id","transform_name","transform_level"], dropna=False)
            for (tid, tname, lvl), df_g in g2:
                rows.append({
                    "scope": "attack_level",
                    "transform_id": int(tid) if pd.notna(tid) else None,
                    "transform_name": tname,
                    "transform_level": lvl,
                    "n_images": int(len(df_g)),
                })
        mdf = pd.DataFrame(rows)
        mp = out["metrics_output"]["path"]
        mc = out["metrics_output"].get("compression","snappy")
        _ensure_dir(os.path.dirname(mp))
        if out["metrics_output"].get("type","parquet") == "csv": mdf.to_csv(mp, index=False)
        else: mdf.to_parquet(mp, index=False, compression=mc)

    return out["table_output"]["path"]
