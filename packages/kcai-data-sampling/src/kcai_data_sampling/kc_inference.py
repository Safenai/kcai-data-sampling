# TODO : ajouter les champs de sorties avec juste le uuid de l'image et les résultats des différents modèles.
from __future__ import annotations

import io
import os
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import yaml

from augment_run import _to_tensor_for_model, _normalize_tensor


def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _np_from_pil(img: Image.Image) -> np.ndarray:
    return np.array(img)


def _read_image_from_bytes(b: Any) -> np.ndarray:
    """Read image from bytes, handling different encodings."""
    if isinstance(b, str):
        try:
            b = b.encode("latin1")
        except Exception:
            import base64
            b = base64.b64decode(b)
    img = _pil_from_bytes(b)
    return _np_from_pil(img)


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    model_name = os.path.basename(ckpt_path)
    
    # Configure quantization backend for CPU
    if device.type == "cpu":
        supported = torch.backends.quantized.supported_engines
        if "qnnpack" in supported:
            torch.backends.quantized.engine = "qnnpack"
        elif "fbgemm" in supported:
            torch.backends.quantized.engine = "fbgemm"
    
    model = None
    try:
        model = torch.jit.load(ckpt_path, map_location=device)
        print(f" TorchScript model loaded: {model_name}")
    except RuntimeError:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            arch = checkpoint.get("arch", "resnet18")
            num_classes = checkpoint.get("num_classes", 1000)
            if arch == "resnet18":
                model = models.resnet18(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise ValueError(f"Unsupported architecture: {arch}")
        else:
            model = checkpoint
        print(f"✅ Checkpoint model loaded: {model_name}")
    
    if hasattr(model, "eval"):
        model.eval()
    model = model.to(device)
    
    return model


def run_inference_on_image(
    model: torch.nn.Module,
    img_arr: np.ndarray,
    device: torch.device,
    mean: List[float],
    std: List[float],
    resize: Optional[Tuple[int, int]],
    output_mode: str = "label_only",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run inference on a single image.
    
    Args:
        output_mode: 
            - "label_only": just label
            - "label_prob": label + prob
            - "top_k": label + prob + top_k
            - "all_probs": label + prob + all_probs
            - "logits": label + prob + logits
            - "extended": label + prob + all_probs + logits (no top_k)
            - "full": everything (label + prob + top_k + all_probs + logits)
        top_k: Number of top predictions to return (for "top_k" and "full" modes)
    
    Returns:
        Dictionary with requested outputs
    """
    t = _to_tensor_for_model(img_arr, resize)
    t = t.unsqueeze(0).to(device)
    t = _normalize_tensor(t, mean, std)
    
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)
        pred_label = logits.argmax(1).item()
        pred_prob = probs[0, pred_label].item()
    
    result: Dict[str, Any] = {}
    
    if output_mode == "label_only":
        result["label"] = int(pred_label)
    
    elif output_mode == "label_prob":
        result["label"] = int(pred_label)
        result["prob"] = float(pred_prob)
    
    elif output_mode == "top_k":
        topk_probs, topk_indices = torch.topk(probs[0], k=min(top_k, probs.shape[1]))
        result["label"] = int(pred_label)
        result["prob"] = float(pred_prob)
        result["top_k_labels"] = topk_indices.cpu().numpy().tolist()
        result["top_k_probs"] = topk_probs.cpu().numpy().tolist()
    
    elif output_mode == "all_probs":
        result["label"] = int(pred_label)
        result["prob"] = float(pred_prob)
        result["all_probs"] = probs[0].cpu().numpy().tolist()
    
    elif output_mode == "logits":
        result["label"] = int(pred_label)
        result["prob"] = float(pred_prob)
        result["logits"] = logits[0].cpu().numpy().tolist()
    
    elif output_mode == "extended":
        # Return label, prob, all_probs, logits (NO top_k)
        result["label"] = int(pred_label)
        result["prob"] = float(pred_prob)
        result["all_probs"] = probs[0].cpu().numpy().tolist()
        result["logits"] = logits[0].cpu().numpy().tolist()
    
    
    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")
    
    return result


def add_inference_columns(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add inference columns to dataframe based on config.
    
    Config structure:
    {
        "image_column": "image_bytes",
        "model": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "resize": [224, 224]
        },
        "output_config": {
            "mode": "label_prob",  # or "top_k", "all_probs", "logits", "extended", "full"
            "top_k": 5,
            
            # NEW: Column selection from input dataframe
            "output_columns": {
                "mode": "all",  # "all" (keep all), "select" (keep only these), "exclude" (remove these)
                "columns": ["sample_uuid", "parent_sample_uuid", "class_id"]  # list of columns
            },
            
            # Custom names for inference columns
            "columns": {
                "label": "pred_label",
                "prob": "pred_prob",
                "top_k_labels": "top5_labels",
                "top_k_probs": "top5_probs",
                "all_probs": "class_probs",
                "logits": "class_logits"
            }
        }
    }
    """
    image_col = config.get("image_column", "image_bytes")
    model_cfg = config.get("model", {})
    output_cfg = config.get("output_config", {})
    
    mean = model_cfg.get("mean", [0.5, 0.5, 0.5])
    std = model_cfg.get("std", [0.5, 0.5, 0.5])
    resize_cfg = model_cfg.get("resize")
    resize = tuple(resize_cfg) if resize_cfg else None
    
    output_mode = output_cfg.get("mode", "label_prob")
    top_k = output_cfg.get("top_k", 5)
    column_names = output_cfg.get("columns", {})
    
    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in dataframe")
    
    print(f"\n{'='*60}")
    print(f"Running inference on {len(df)} images")
    print(f"Output mode: {output_mode}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Collect results
    results: List[Dict[str, Any]] = []
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        try:
            img_arr = _read_image_from_bytes(row[image_col])
            result = run_inference_on_image(
                model, img_arr, device, mean, std, resize, output_mode, top_k
            )
            results.append(result)
        except Exception as e:
            if failed_count == 0:
                print(f"\n⚠️  Failed to process row {idx}: {e}")
            results.append({})
            failed_count += 1
    
    # Add columns to dataframe with custom names
    # Handle column selection from config
    output_columns_cfg = output_cfg.get("output_columns", {})
    keep_mode = output_columns_cfg.get("mode", "all")  # "all", "select", "exclude"
    columns_list = output_columns_cfg.get("columns", [])
    
    # Start with selected input columns based on mode
    if keep_mode == "select":
        # Keep only specified columns from input
        missing_cols = [c for c in columns_list if c not in df.columns]
        if missing_cols:
            print(f"\n⚠️  Warning: Columns not found in input: {missing_cols}")
        available_cols = [c for c in columns_list if c in df.columns]
        df_out = df[available_cols].copy()
        print(f"\n📋 Keeping {len(available_cols)} selected columns: {available_cols}")
    elif keep_mode == "exclude":
        # Keep all columns except specified ones
        cols_to_keep = [c for c in df.columns if c not in columns_list]
        df_out = df[cols_to_keep].copy()
        print(f"\n📋 Excluding {len(columns_list)} columns, keeping {len(cols_to_keep)}")
    else:  # "all" (default)
        # Keep all input columns
        df_out = df.copy()
        print(f"\n📋 Keeping all {len(df.columns)} input columns")
    
    # Determine which columns to add based on output_mode
    keys_to_add: List[str] = []
    if output_mode == "label_only":
        keys_to_add = ["label"]
    elif output_mode == "label_prob":
        keys_to_add = ["label", "prob"]
    elif output_mode == "top_k":
        keys_to_add = ["label", "prob", "top_k_labels", "top_k_probs"]
    elif output_mode == "all_probs":
        keys_to_add = ["label", "prob", "all_probs"]
    elif output_mode == "logits":
        keys_to_add = ["label", "prob", "logits"]
    elif output_mode == "extended":
        keys_to_add = ["label", "prob", "all_probs", "logits"]
    elif output_mode == "full":
        keys_to_add = ["label", "prob", "top_k_labels", "top_k_probs", "all_probs", "logits"]
    
    for key in keys_to_add:
        col_name = column_names.get(key, key)  # Use custom name or default
        df_out[col_name] = [r.get(key) for r in results]
    
    success_count = len(df) - failed_count
    print(f"\n✅ Inference completed:")
    print(f"   - Successful: {success_count}/{len(df)}")
    print(f"   - Failed: {failed_count}/{len(df)}")
    print(f"   - Added columns: {[column_names.get(k, k) for k in keys_to_add]}")
    
    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on parquet file and add prediction columns with custom names"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify the input file in-place (overwrites it with added columns)",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    inference_cfg = config.get("inference", {})
    
    # Paths
    input_path = inference_cfg.get("input")
    output_path = inference_cfg.get("output")
    model_path = inference_cfg["model"]["ckpt_path"]
    
    if not input_path:
        raise ValueError("Config must specify 'inference.input'")
    
    # Handle in-place mode
    if args.in_place:
        output_path = input_path
        print(f" IN-PLACE MODE: Will overwrite {input_path}")
    elif not output_path:
        raise ValueError("Config must specify 'inference.output' or use --in-place flag")
    elif input_path == output_path:
        print(f"IN-PLACE MODE: input == output, will overwrite {input_path}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)
    
    # Load input parquet
    print(f"\nLoading parquet from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Run inference
    df_out = add_inference_columns(df, model, device, inference_cfg)
    
    # Save output
    print(f"\nSaving to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_parquet(output_path, index=False, compression="snappy")
    
    print("\nDone!")
    print(f"Output columns: {list(df_out.columns)}")


if __name__ == "__main__":
    main()

