# splitters/hash.py
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from .base import Splitter

def _hash16(buf: pa.Buffer) -> str:
    """generate a SHA256 hash truncated to 16 characters."""
    return hashlib.sha256(buf.to_pybytes()).hexdigest()[:16]

def _frac_from_hex(h: str) -> float:
    return int(h, 16) / (16 ** len(h))

class HashSplitter(Splitter):
    """ deterministic splitter based on image hashes. Non-stratified - class distribution may vary between splits.
    """
    
    def __init__(self, image_bytes_col="image_bytes", ratios=(0.7, 0.2, 0.1), out_col="split"):
        """
        Args:
            image_bytes_col: Name of the column containing image bytes
            ratios: Tuple (train_ratio, val_ratio, test_ratio) - must sum to 1.0
            out_col: Name of the output column for splits
        """
        assert abs(sum(ratios) - 1.0) < 1e-9, f"Ratios must sum to 1.0, got {sum(ratios)}"
        self.image_bytes_col = image_bytes_col
        self.ratios = ratios
        self.out_col = out_col

    def apply(self, table: pa.Table) -> pa.Table:

        tbl = table
        
        if "image_hash" not in tbl.column_names:
            hashes = [_hash16(tbl[self.image_bytes_col][i].as_buffer()) for i in range(len(tbl))]
            tbl = tbl.append_column("image_hash", pa.array(hashes, type=pa.string()))

        xs = np.array([_frac_from_hex(h) for h in tbl["image_hash"].to_pylist()], dtype=np.float64)
        
        thr1, thr2 = self.ratios[0], self.ratios[0] + self.ratios[1]
        split = np.where(xs < thr1, "train", np.where(xs < thr2, "val", "test"))
        
        return tbl.append_column(self.out_col, pa.array(split, type=pa.string()))
